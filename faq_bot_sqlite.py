import os
import sqlite3
import json
from glob import glob
from pprint import pprint
from typing import List, Dict, Optional, Literal, Generator
from dataclasses import dataclass
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# 1. 환경변수 로드
load_dotenv()

# 2. Pydantic 및 데이터 모델 정의
class MetadataFilter(BaseModel):
    """Chroma DB 메타데이터 필터 조건"""
    keyword: Optional[str] = Field(default=None, description="검색할 키워드")
    keyword_operator: Optional[Literal["$eq", "$ne"]] = Field(
        default=None, description="키워드 비교 연산자"
    )
    question_id_min: Optional[int] = Field(default=None, description="질문 ID 최소값")
    question_id_min_operator: Optional[Literal["$gt", "$gte"]] = Field(
        default=None, description="최소값 연산자 ($gt: 초과, $gte: 이상)"
    )
    question_id_max: Optional[int] = Field(default=None, description="질문 ID 최대값")
    question_id_max_operator: Optional[Literal["$lt", "$lte"]] = Field(
        default=None, description="최대값 연산자 ($lt: 미만, $lte: 이하)"
    )
    logical_operator: Optional[Literal["$and", "$or"]] = Field(
        default="$and", description="복합 조건 결합 연산자"
    )

@dataclass
class SearchResult:
    context: str
    source_documents: Optional[List[Document]]

# 3. SQLite 트리밍 & 요약 영구 저장 히스토리 클래스
class SQLiteTrimmedAndSummarizedHistory(BaseChatMessageHistory):
    """
    SQLite를 사용하여 메시지 트리밍과 대화 요약을 영구 저장하는 대화 히스토리 클래스
    """
    def __init__(
        self, 
        session_id: str, 
        db_path: str = "chat_history_advanced.db", 
        max_messages: int = 4,              # 트리밍 기준 (유지할 상세 대화 수)
        llm: Optional[ChatOpenAI] = None
    ):
        self.session_id = session_id
        self.db_path = db_path
        self.max_messages = max_messages
        # gpt-4.1-mini 사용하되 없으면 gpt-4o-mini로 자동 매칭되도록 구성
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        self._create_tables()

    def _create_tables(self):
        """SQLite 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 상세 대화 저장용 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_type TEXT,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 세션별 대화 요약본 저장용 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get_summary(self) -> str:
        """현재 세션의 요약본 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT summary FROM summaries WHERE session_id = ?", (self.session_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else ""

    def save_summary(self, summary: str):
        """요약본 저장/업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO summaries (session_id, summary, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET 
                summary = excluded.summary,
                updated_at = CURRENT_TIMESTAMP
        """, (self.session_id, summary))
        conn.commit()
        conn.close()

    @property
    def messages(self) -> List[BaseMessage]:
        """저장된 최근 상세 메시지 목록 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message_type, content, metadata
            FROM messages 
            WHERE session_id = ?
            ORDER BY id ASC
        """, (self.session_id,))
        
        loaded_messages = []
        for row in cursor.fetchall():
            msg_type, content, metadata = row
            if msg_type == "HumanMessage":
                msg = HumanMessage(content=content)
            elif msg_type == "AIMessage":
                msg = AIMessage(content=content)
            elif msg_type == "SystemMessage":
                msg = SystemMessage(content=content)
            else:
                continue
                
            if metadata:
                msg.additional_kwargs = json.loads(metadata)
            loaded_messages.append(msg)
            
        conn.close()
        return loaded_messages

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """메시지를 추가하고, 한도를 초과하면 요약을 생성한 뒤 트리밍합니다."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for msg in messages:
            cursor.execute("""
                INSERT INTO messages (session_id, message_type, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                self.session_id,
                msg.__class__.__name__,
                msg.content,
                json.dumps(msg.additional_kwargs)
            ))
        conn.commit()
        
        # 현재 저장된 전체 대화 수 확인
        cursor.execute("SELECT id, message_type, content FROM messages WHERE session_id = ? ORDER BY id ASC", (self.session_id,))
        all_rows = cursor.fetchall()
        conn.close()
        
        # 임계치 초과 시 트리밍 및 요약 진행
        if len(all_rows) > self.max_messages:
            num_to_delete = len(all_rows) - self.max_messages
            to_delete_rows = all_rows[:num_to_delete]
            
            # 요약할 텍스트 추출
            delete_chats = []
            for _, msg_type, content in to_delete_rows:
                role = "User" if msg_type == "HumanMessage" else "AI"
                delete_chats.append(f"{role}: {content}")
            new_chats_to_summarize = "\n".join(delete_chats)
            
            # 기존 요약 가져오기
            existing_summary = self.get_summary()
            
            # LLM을 사용해 요약 업데이트
            summary_prompt = (
                f"기존 요약본:\n{existing_summary or '없음'}\n\n"
                f"새로 추가된 대화:\n{new_chats_to_summarize}\n\n"
                "위 두 내용을 바탕으로 전체 맥락이 이어지도록 간결한 통합 요약(한글)을 작성해 주세요."
            )
            
            summary_msg = self.llm.invoke([
                SystemMessage(content="당신은 대화의 맥락을 요약하여 기록하는 비서입니다."),
                HumanMessage(content=summary_prompt)
            ])
            
            # 새 요약본 DB 저장
            self.save_summary(summary_msg.content)
            print(f"📊 [SQLite 요약 저장]: {summary_msg.content}")
            
            # DB에서 삭제 대상 메시지 지우기 (트리밍)
            delete_ids = [row[0] for row in to_delete_rows]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM messages WHERE id IN ({','.join(['?'] * len(delete_ids))})",
                delete_ids
            )
            conn.commit()
            conn.close()
            print(f"✂️ [상세 메시지 {len(delete_ids)}개 트리밍 완료]")

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        cursor.execute("DELETE FROM summaries WHERE session_id = ?", (self.session_id,))
        conn.commit()
        conn.close()

# 4. RAG 시스템 구현
class HousingRAGSystem:
    def __init__(
        self, 
        llm: ChatOpenAI, 
        eval_llm: ChatOpenAI,
        retriever
    ):
        self.llm = llm
        self.eval_llm = eval_llm
        self.retriever = retriever
        self.vectorstore = retriever.vectorstore
        
        # 메타데이터 필터 추출기 설정
        self.filter_extractor = self.eval_llm.with_structured_output(MetadataFilter)
        self.filter_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                 사용자 쿼리에서 Chroma DB 검색 필터 조건을 추출합니다.
                 정보가 없으면 null을 반환하세요.
            """),
            ("human", "{question}")
        ])

    def _build_chroma_filter(self, filter_model: MetadataFilter) -> Optional[Dict]:
        conditions = []
        if filter_model.keyword:
            conditions.append({"keyword": {filter_model.keyword_operator: filter_model.keyword}})
        if filter_model.question_id_min is not None:
            conditions.append({"question_id": {filter_model.question_id_min_operator: filter_model.question_id_min}})
        if filter_model.question_id_max is not None:
            conditions.append({"question_id": {filter_model.question_id_max_operator: filter_model.question_id_max}})
            
        if not conditions: return None
        return conditions[0] if len(conditions) == 1 else {filter_model.logical_operator: conditions}

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_source_documents(self, docs: Optional[List[Document]]) -> str:
        if not docs:
            return "\n\nℹ️ 관련 문서를 찾을 수 없습니다."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source_info = []
            
            if 'question_id' in metadata:
                source_info.append(f"ID: {metadata['question_id']}")
            if 'keyword' in metadata:
                source_info.append(f"키워드: {metadata['keyword']}")
            if 'chapter' in metadata:
                source_info.append(f"주제: {metadata['chapter']}")
                
            formatted_docs.append(
                f"**[참조 문서 {i}]** {' | '.join(source_info) if source_info else '출처 정보 없음'}\n"
                f"> {doc.page_content.replace('\n', ' ')}"
            )
        return "\n\n**[근거 문서]**\n" + "\n\n".join(formatted_docs)

    def _check_relevance(self, docs: List[Document], question: str) -> List[Document]:
        relevant_docs = []
        if not docs: return relevant_docs
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "주어진 컨텍스트가 질문에 답변하는데 필요한 정보를 포함하고 있는지 평가하세요.\n"
                       "직접적으로 포함하거나 논리적으로 추론 가능하다면 'Yes', 아니면 'No'로만 답변하세요."),
            ("human", "<context>\n{context}</context>\n\n<question>\n{question}</question>")
        ])
        chain = prompt | self.eval_llm | StrOutputParser()

        print("\n🔍 [관련성 평가 시작]")
        for doc in docs:
            result = chain.invoke({"context": doc.page_content, "question": question}).lower()
            if "yes" in result:
                relevant_docs.append(doc)
            else:
                print(f"  - 문서 배제됨: {doc.page_content[:30]}...")
                
        return relevant_docs

    def search_documents(self, question: str) -> SearchResult:
        try:
            extracted_filter = self.filter_extractor.invoke(self.filter_prompt.format(question=question))
            chroma_filter = self._build_chroma_filter(extracted_filter)
            print(f"🎯 [적용된 필터]: {chroma_filter}")
            
            search_kwargs = {"k": 4}
            if chroma_filter: search_kwargs["filter"] = chroma_filter
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            
            raw_docs = retriever.invoke(question)
            print(f"📄 [1차 검색 문서 개수]: {len(raw_docs)}")
            
            relevant_docs = self._check_relevance(raw_docs, question) 
            print(f"✅ [최종 유효 문서 개수]: {len(relevant_docs)}")
            
            return SearchResult(
                context=self._format_docs(relevant_docs) if relevant_docs else "관련 문서를 찾을 수 없습니다.",
                source_documents=relevant_docs,
            )

        except Exception as e:
            print(f"문서 검색 중 오류 발생: {e}")
            return SearchResult(context="문서 검색 중 오류가 발생했습니다.", source_documents=None)

    def generate_answer(self, message: str, history: List) -> Generator[str, None, None]:
        # 세션 관리를 위한 고정 ID (Gradio 단일 사용자 데모용)
        session_id = "gradio_faq_session"
        
        # 1. SQLite 기반의 Trimmed & Summarized History 인스턴스 조회
        db_history = SQLiteTrimmedAndSummarizedHistory(
            session_id=session_id,
            max_messages=4,
            llm=self.llm
        )
        
        # 2. 문서 검색 실행
        search_result = self.search_documents(message)
        if not search_result.source_documents:
            yield "⚠️ 죄송합니다. 제공된 FAQ 문서 내에서 질문에 대한 명확한 답변이나 근거를 찾을 수 없습니다. 더 구체적으로 질문해 주시거나 청약홈을 확인해 주세요."
            return

        # 3. 데이터베이스에서 기존 누적 요약본 가져오기
        existing_summary = db_history.get_summary()

        # 4. 프롬프트 메시지 목록 구성
        messages = []
        system_content = """당신은 주택청약 전문 상담가입니다. 다음 지침을 엄격히 따르세요:
1. 제공된 [문서들]의 내용만을 기반으로 답변하세요.
2. 문서에 명확한 근거가 없는 내용은 "근거 없음"이라고 답변하세요.
3. 추측이나 일반적인 사전 지식을 섞어서 사용하지 마세요."""

        if existing_summary:
            system_content += f"\n\n[이전 대화 요약]\n{existing_summary}"
            
        messages.append(SystemMessage(content=system_content))

        # DB에 저장되어 있던 최신 상세 메시지들을 추가
        messages.extend(db_history.messages)

        # 현재 질문과 컨텍스트 추가
        current_human_content = f"<context>\n{search_result.context}\n</context>\n\n<question>{message}</question>"
        messages.append(HumanMessage(content=current_human_content))

        # 5. 답변 생성 및 스트리밍
        full_answer = ""
        try:
            for chunk in self.llm.stream(messages):
                full_answer += chunk.content
                yield full_answer
            
            # 출처(Reference) 표시 추가
            sources = self._format_source_documents(search_result.source_documents)
            final_response = f"{full_answer}\n\n---\n{sources}"
            yield final_response

            # 6. 대화 기록을 SQLite에 업데이트 및 트리밍 유도
            db_history.add_messages([
                HumanMessage(content=message),
                AIMessage(content=full_answer)
            ])
            
        except Exception as e:
            yield f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# 5. 실행 엔트리 포인트
if __name__ == "__main__":
    print("🚀 스마트 주택청약 FAQ 챗봇을 실행합니다...")
    
    # 임베딩 및 벡터스토어 로드
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="housing_faq_db",
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
    )
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"fetch_k": 10, "k": 3, "lambda_mult": 0.5}
    )

    # LLM 초기화 (gpt-4.1-mini 사용하되 없으면 gpt-4o-mini로 자동 셋업)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, top_p=0.9)
    eval_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

    rag_system = HousingRAGSystem(
        llm=llm,
        eval_llm=eval_llm,
        retriever=mmr_retriever
    )

    # Gradio UI 구성
    demo = gr.ChatInterface(
        fn=rag_system.generate_answer,
        title="🏢 스마트 주택청약 FAQ 챗봇 (SQLite)",
        description="SQLite 대화 이력 관리(트리밍 & 요약)가 탑재된 주택청약 RAG 시스템입니다.",
        examples=[
            "신혼부부 특별공급 혼인기간 기준이 어떻게 되나요?",
            "청약통장 가입 연령에 제한이 있나요?",
            "청약신청 지역을 판단하는 기준은 무엇인가요?"
        ],
        fill_height=True
    )
    
    demo.launch(share=False)
