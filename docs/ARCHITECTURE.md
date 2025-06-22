# System Architecture

This document provides a comprehensive overview of BasicChat's technical architecture, including component interactions, data flows, and system design principles.

[← Back to README](../README.md)

---

## 🏗️ High-Level Architecture

BasicChat follows a **layered microservices architecture** with clear separation of concerns, enabling scalability, maintainability, and testability. The system is built around a **reasoning engine** that orchestrates multiple specialized components.

```mermaid
graph TB
    subgraph "🎨 User Interface"
        UI[Web Interface]
        AUDIO[Audio Processing]
    end
    
    subgraph "🧠 Core Logic"
        RE[Reasoning Engine]
        DP[Document Processor]
        TR[Tool Registry]
    end
    
    subgraph "⚡ Services"
        AO[Ollama Client]
        VS[Vector Store]
        CS[Cache Service]
        WS[Web Search]
    end
    
    subgraph "🗄️ Storage"
        CHROMA[Vector Database]
        CACHE[Memory Cache]
        FILES[File Storage]
    end
    
    subgraph "🌐 External"
        OLLAMA[LLM Server]
        DDG[Search Engine]
    end
    
    %% User Interface Connections
    UI --> RE
    UI --> DP
    AUDIO --> RE
    
    %% Core Logic Connections
    RE --> AO
    RE --> VS
    RE --> TR
    DP --> VS
    TR --> WS
    
    %% Service Connections
    AO --> OLLAMA
    VS --> CHROMA
    CS --> CACHE
    WS --> DDG
    
    %% Storage Connections
    CHROMA --> FILES
    CACHE --> FILES
    
    %% Styling
    classDef ui fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef core fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C
    classDef service fill:#E8F5E8,stroke:#388E3C,stroke-width:2px,color:#1B5E20
    classDef storage fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#E65100
    classDef external fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F
    
    class UI,AUDIO ui
    class RE,DP,TR core
    class AO,VS,CS,WS service
    class CHROMA,CACHE,FILES storage
    class OLLAMA,DDG external
```

## 🧩 Core Components

### 1. **Reasoning Engine** (`reasoning_engine.py`)
The central orchestrator that manages different reasoning strategies and coordinates tool usage.

**Key Responsibilities:**
- **Multi-Modal Reasoning**: Supports 5 reasoning modes (Auto, Standard, Chain-of-Thought, Multi-Step, Agent-Based)
- **Tool Orchestration**: Intelligently selects and executes appropriate tools
- **Context Management**: Integrates document context with user queries
- **Response Synthesis**: Combines multiple sources into coherent answers

**Architecture Pattern:** Strategy Pattern with Factory Method

```mermaid
classDiagram
    class ReasoningEngine {
        +run(query, mode, document_processor) ReasoningResult
        -_retrieve_and_format_context() str
    }
    
    class ReasoningAgent {
        +run(query, context, stream_callback) ReasoningResult
        -_create_tools() List[Tool]
        -_web_search(query) str
        -_enhanced_calculate(expression) str
    }
    
    class ReasoningChain {
        +execute_reasoning(question, context) ReasoningResult
        -_parse_chain_of_thought_response() tuple
    }
    
    class MultiStepReasoning {
        +step_by_step_reasoning(query, context) ReasoningResult
        -_generate_sub_queries() List[str]
        -_synthesize_final_answer() str
    }
    
    class AutoReasoning {
        +auto_reason(query, context) ReasoningResult
        -_analyze_query_complexity() Dict
        -_fallback_analysis() Dict
    }
    
    ReasoningEngine --> ReasoningAgent
    ReasoningEngine --> ReasoningChain
    ReasoningEngine --> MultiStepReasoning
    ReasoningEngine --> AutoReasoning
```

### 2. **Document Processor** (`document_processor.py`)
Manages the complete document lifecycle with advanced RAG capabilities.

**Key Features:**
- **Multi-Format Support**: PDF, text, markdown, images (OCR)
- **Intelligent Chunking**: Recursive character splitting with overlap
- **Vector Embeddings**: Local embedding generation with Ollama
- **Semantic Search**: ChromaDB-based similarity search
- **Memory Management**: Automatic cleanup and resource optimization

```mermaid
graph LR
    subgraph "📄 Document Processing"
        UPLOAD[File Upload]
        EXTRACT[Text Extraction]
        CHUNK[Text Chunking]
        EMBED[Vector Embedding]
        STORE[Vector Storage]
        SEARCH[Semantic Search]
    end
    
    subgraph "🖼️ Image Processing"
        IMG[Image Upload]
        OCR[Vision Model OCR]
        DESC[Description Generation]
    end
    
    UPLOAD --> EXTRACT
    EXTRACT --> CHUNK
    CHUNK --> EMBED
    EMBED --> STORE
    STORE --> SEARCH
    
    IMG --> OCR
    OCR --> DESC
    DESC --> CHUNK
    
    classDef pipeline fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef image fill:#E8F5E8,stroke:#388E3C,stroke-width:2px,color:#1B5E20
    
    class UPLOAD,EXTRACT,CHUNK,EMBED,STORE,SEARCH pipeline
    class IMG,OCR,DESC image
```

### 3. **Async Ollama Client** (`utils/async_ollama.py`)
High-performance client for Ollama API with advanced connection management.

**Performance Features:**
- **Connection Pooling**: 100 total connections, 30 per host
- **Rate Limiting**: Token bucket algorithm (10 req/sec default)
- **Retry Logic**: Exponential backoff with 3 attempts
- **Streaming Support**: Real-time response streaming
- **Health Monitoring**: Automatic service availability checks

```mermaid
sequenceDiagram
    participant Client as AsyncOllamaClient
    participant Pool as Connection Pool
    participant Throttle as Rate Limiter
    participant Ollama as Ollama Server
    participant Cache as Response Cache
    
    Client->>Pool: Get Connection
    Pool-->>Client: Available Connection
    
    Client->>Throttle: Check Rate Limit
    Throttle-->>Client: Allow Request
    
    Client->>Cache: Check Cache
    alt Cache Hit
        Cache-->>Client: Cached Response
    else Cache Miss
        Client->>Ollama: Make Request
        Ollama-->>Client: Response
        Client->>Cache: Store Response
    end
    
    Client->>Pool: Return Connection
```

### 4. **Tool Registry** (`utils/enhanced_tools.py`)
Extensible tool system providing specialized capabilities.

**Available Tools:**
- **Enhanced Calculator**: Advanced mathematical operations with step-by-step reasoning
- **Time Tools**: Timezone-aware time calculations and conversions
- **Web Search**: Real-time information retrieval via DuckDuckGo
- **Document Summary**: Intelligent document summarization

```mermaid
graph TD
    subgraph "🛠️ Tool Registry"
        TR[Tool Registry]
        CALC[Enhanced Calculator]
        TIME[Time Tools]
        WEB[Web Search]
        DOC[Document Summary]
    end
    
    subgraph "🔧 Tool Capabilities"
        MATH[Mathematical Operations]
        TZ[Timezone Conversions]
        SEARCH[Real-time Search]
        SUMMARY[Document Analysis]
    end
    
    TR --> CALC
    TR --> TIME
    TR --> WEB
    TR --> DOC
    
    CALC --> MATH
    TIME --> TZ
    WEB --> SEARCH
    DOC --> SUMMARY
```

## 🔄 Data Flow Architecture

### Standard Query Processing

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant RE as Reasoning Engine
    participant TR as Tool Registry
    participant AO as Async Ollama
    participant Cache as Cache Service
    
    User->>UI: Submit Query
    UI->>RE: Process Query
    
    RE->>Cache: Check Cache
    alt Cache Hit
        Cache-->>RE: Cached Response
        RE-->>UI: Return Response
    else Cache Miss
        RE->>RE: Analyze Query Complexity
        RE->>TR: Select Appropriate Tools
        
        alt Tool Required
            RE->>TR: Execute Tool
            TR-->>RE: Tool Result
        end
        
        RE->>AO: Generate LLM Response
        AO-->>RE: LLM Response
        RE->>Cache: Store Response
        RE-->>UI: Return Response
    end
    
    UI-->>User: Display Result
```

### Document Analysis (RAG) Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant DP as Document Processor
    participant VS as Vector Store
    participant RE as Reasoning Engine
    participant AO as Async Ollama
    
    User->>UI: Upload Document
    UI->>DP: Process Document
    
    DP->>DP: Extract Text
    DP->>DP: Create Chunks
    DP->>DP: Generate Embeddings
    DP->>VS: Store Vectors
    VS-->>DP: Confirm Storage
    DP-->>UI: Processing Complete
    
    User->>UI: Ask Question
    UI->>RE: Process Query
    
    RE->>VS: Search Relevant Chunks
    VS-->>RE: Top K Chunks
    RE->>AO: Generate Answer with Context
    AO-->>RE: Contextual Response
    RE-->>UI: Return Answer
    UI-->>User: Display Answer
```

## 🚀 Performance Architecture

### Caching Strategy

```mermaid
graph TB
    subgraph "💾 Multi-Layer Cache"
        L1[L1: Memory Cache]
        L2[L2: Redis Cache]
        L3[L3: Disk Cache]
    end
    
    subgraph "🔑 Cache Keys"
        QUERY[Query Hash]
        MODEL[Model Name]
        PARAMS[Parameters]
        CONTEXT[Context Hash]
    end
    
    subgraph "📊 Cache Performance"
        HIT_RATE[70-85% Hit Rate]
        RESPONSE_TIME[50-80% Faster]
        TTL[Configurable TTL]
    end
    
    QUERY --> L1
    MODEL --> L1
    PARAMS --> L1
    CONTEXT --> L1
    
    L1 --> L2
    L2 --> L3
    
    L1 --> HIT_RATE
    L2 --> RESPONSE_TIME
    L3 --> TTL
```

### Async Processing Pipeline

```mermaid
graph LR
    subgraph "⚡ Async Processing"
        REQ[Request Queue]
        WORKER1[Worker 1]
        WORKER2[Worker 2]
        WORKER3[Worker 3]
        RESP[Response Queue]
    end
    
    subgraph "🔧 Connection Pool"
        POOL[Connection Pool]
        LIMITER[Rate Limiter]
        RETRY[Retry Logic]
    end
    
    REQ --> WORKER1
    REQ --> WORKER2
    REQ --> WORKER3
    
    WORKER1 --> POOL
    WORKER2 --> POOL
    WORKER3 --> POOL
    
    POOL --> LIMITER
    LIMITER --> RETRY
    RETRY --> RESP
```

## 🔒 Security & Privacy Architecture

### Data Privacy Model

```mermaid
graph TB
    subgraph "🔒 Privacy Controls"
        LOCAL[Local Processing Only]
        NO_EXTERNAL[No External APIs]
        ENCRYPT[Encrypted Storage]
        CLEANUP[Auto Cleanup]
    end
    
    subgraph "🛡️ Security Measures"
        VALIDATION[Input Validation]
        SANITIZATION[Expression Sanitization]
        RATE_LIMIT[Rate Limiting]
        ERROR_HANDLING[Error Handling]
    end
    
    subgraph "📊 Data Flow"
        USER[User Input]
        PROCESS[Local Processing]
        STORE[Local Storage]
        CLEAN[Auto Cleanup]
    end
    
    USER --> VALIDATION
    VALIDATION --> SANITIZATION
    SANITIZATION --> PROCESS
    
    PROCESS --> LOCAL
    PROCESS --> NO_EXTERNAL
    
    PROCESS --> STORE
    STORE --> ENCRYPT
    STORE --> CLEANUP
    CLEANUP --> CLEAN
```

## 🏗️ Technology Stack

### Core Technologies

```mermaid
graph TB
    subgraph "🐍 Backend"
        PYTHON[Python 3.11+]
        STREAMLIT[Streamlit]
        LANGCHAIN[LangChain]
        PYDANTIC[Pydantic]
    end
    
    subgraph "🤖 AI/ML"
        OLLAMA[Ollama]
        CHROMA[ChromaDB]
        EMBEDDINGS[Text Embeddings]
        VISION[Vision Models]
    end
    
    subgraph "⚡ Performance"
        AIOHTTP[aiohttp]
        REDIS[Redis]
        CACHETOOLS[cachetools]
        THROTTLE[asyncio-throttle]
    end
    
    subgraph "🛠️ Tools"
        DUCKDUCKGO[DuckDuckGo Search]
        GTTS[gTTS]
        PILLOW[Pillow]
        PYTZ[pytz]
    end
    
    PYTHON --> STREAMLIT
    PYTHON --> LANGCHAIN
    PYTHON --> PYDANTIC
    
    LANGCHAIN --> OLLAMA
    LANGCHAIN --> CHROMA
    LANGCHAIN --> EMBEDDINGS
    LANGCHAIN --> VISION
    
    STREAMLIT --> AIOHTTP
    STREAMLIT --> REDIS
    STREAMLIT --> CACHETOOLS
    STREAMLIT --> THROTTLE
    
    LANGCHAIN --> DUCKDUCKGO
    STREAMLIT --> GTTS
    STREAMLIT --> PILLOW
    STREAMLIT --> PYTZ
```

## 📈 Scalability Considerations

### Horizontal Scaling

```mermaid
graph TB
    subgraph "🔄 Load Balancer"
        LB[Load Balancer]
    end
    
    subgraph "🖥️ Application Instances"
        INST1[Instance 1]
        INST2[Instance 2]
        INST3[Instance 3]
    end
    
    subgraph "🗄️ Shared Services"
        REDIS_SHARED[Redis Cluster]
        CHROMA_SHARED[ChromaDB Cluster]
    end
    
    subgraph "🔍 Health Checks"
        HEALTH[Health Monitor]
        FAILOVER[Auto Failover]
    end
    
    LB --> INST1
    LB --> INST2
    LB --> INST3
    
    INST1 --> REDIS_SHARED
    INST2 --> REDIS_SHARED
    INST3 --> REDIS_SHARED
    
    INST1 --> CHROMA_SHARED
    INST2 --> CHROMA_SHARED
    INST3 --> CHROMA_SHARED
    
    HEALTH --> INST1
    HEALTH --> INST2
    HEALTH --> INST3
    
    HEALTH --> FAILOVER
    FAILOVER --> LB
```

## 🔗 Related Documentation

- **[Features Overview](FEATURES.md)** - Detailed feature documentation
- **[Development Guide](DEVELOPMENT.md)** - Contributing and development workflows
- **[Roadmap](ROADMAP.md)** - Future development plans
- **[Reasoning Features](../REASONING_FEATURES.md)** - Advanced reasoning engine details

## 📚 References

### Core Technologies
- **Ollama**: [https://ollama.ai](https://ollama.ai) - Local large language model server
- **Streamlit**: [https://streamlit.io](https://streamlit.io) - Web application framework
- **LangChain**: [https://langchain.com](https://langchain.com) - LLM application framework
- **ChromaDB**: [https://chromadb.ai](https://chromadb.ai) - Vector database

### Research Papers
- **Chain-of-Thought Reasoning**: Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *arXiv preprint arXiv:2201.11903*, 2022.
- **Retrieval-Augmented Generation**: Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 9459-9474.
- **Vector Similarity Search**: Johnson et al. "Billion-Scale Similarity Search with GPUs." *arXiv preprint arXiv:1908.10396*, 2019.

---

[← Back to README](../README.md) | [Features →](FEATURES.md) | [Development →](DEVELOPMENT.md) | [Roadmap →](ROADMAP.md) 