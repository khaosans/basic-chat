# 🏗️ BasicChat System Architecture

> **Comprehensive technical architecture documentation for BasicChat's modular, scalable design**

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture Layers](#architecture-layers)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Performance Architecture](#performance-architecture)
- [Security & Privacy](#security--privacy)
- [Scalability Considerations](#scalability-considerations)
- [Technology Stack](#technology-stack)
- [References](#references)

---

## 🎯 System Overview

BasicChat is built on a **layered microservices architecture** that prioritizes modularity, performance, and privacy. The system is designed to run entirely locally while maintaining enterprise-grade capabilities.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb'}}}%%
graph TB
    subgraph "User Interface"
        UI[Streamlit UI]
        API[REST API]
    end
    
    subgraph "Core Services"
        RE[Reasoning Engine]
        DP[Document Processor]
        TM[Tool Manager]
    end
    
    subgraph "AI Services"
        LLM[LLM Service]
        EMB[Embedding Service]
        VISION[Vision Service]
    end
    
    subgraph "Data Storage"
        VDB[Vector Database]
        CACHE[Cache Store]
        FILES[File Storage]
    end
    
    UI --> RE
    API --> RE
    
    RE --> LLM
    RE --> EMB
    RE --> VISION
    
    DP --> VDB
    TM --> CACHE
    
    LLM --> OLLAMA[Ollama Models]
    EMB --> CHROMADB[ChromaDB]
    VDB --> CHROMADB
    CACHE --> REDIS[Redis Cache]
    
    classDef ui fill:#dbeafe,stroke:#1e3a8a,stroke-width:2px,color:#1f2937
    classDef core fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#1f2937
    classDef ai fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    classDef data fill:#f3e8ff,stroke:#7c3aed,stroke-width:2px,color:#1f2937
    classDef external fill:#fef2f2,stroke:#dc2626,stroke-width:2px,color:#1f2937
    
    class UI,API ui
    class RE,DP,TM core
    class LLM,EMB,VISION ai
    class VDB,CACHE,FILES data
    class OLLAMA,CHROMADB,REDIS external
```

---

## 🏛️ Architecture Layers

### **1. Presentation Layer**

The presentation layer handles all user interactions and provides multiple interfaces for accessing BasicChat's capabilities.

**Components:**
- **Streamlit UI**: Primary web interface with real-time updates
- **REST API**: Programmatic access for integrations

### **2. Application Layer**

The application layer contains the core business logic and orchestrates interactions between services.

**Components:**
- **Reasoning Engine**: Multi-modal reasoning with 5 different modes
- **Document Processor**: Advanced RAG pipeline with intelligent chunking
- **Tool Manager**: Plugin architecture for extensible functionality

### **3. Service Layer**

The service layer provides specialized services for different types of AI processing.

**Components:**
- **LLM Service**: Language model interactions and management
- **Embedding Service**: Vector embeddings for semantic search
- **Vision Service**: Image processing and analysis

### **4. Data Layer**

The data layer manages all data persistence and retrieval operations.

**Components:**
- **Vector Database**: ChromaDB for semantic search and document storage
- **Cache Store**: Redis for high-performance caching
- **File Storage**: Local file system for document and media storage

### **5. External Layer**

The external layer interfaces with external systems and services.

**Components:**
- **Ollama Models**: Local LLM inference engine
- **ChromaDB**: Vector database for embeddings
- **Redis**: In-memory cache for performance

---

## 🔧 Core Components

### **Reasoning Engine**

The reasoning engine is the heart of BasicChat, providing multiple reasoning modes for different use cases.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb', 'edgeLabelBackground': '#f9fafb'}}}%%
graph TB
    subgraph "Reasoning Modes"
        COT[Chain of Thought]
        AGENT[Agent-Based]
        AUTO[Auto Mode]
        MULTI[Multi-Step]
        HYBRID[Hybrid Mode]
    end
    
    subgraph "Core Engine"
        PARSER[Query Parser]
        CONTEXT[Context Manager]
        EXECUTOR[Tool Executor]
        SYNTHESIZER[Response Synthesizer]
    end
    
    subgraph "Tool Integration"
        CALC[Calculator]
        TIME[Time Tools]
        WEB[Web Search]
        DOCS[Document Tools]
        CUSTOM[Custom Tools]
    end
    
    COT --> PARSER
    AGENT --> PARSER
    AUTO --> PARSER
    MULTI --> PARSER
    HYBRID --> PARSER
    
    PARSER --> CONTEXT
    CONTEXT --> EXECUTOR
    EXECUTOR --> SYNTHESIZER
    
    EXECUTOR --> CALC
    EXECUTOR --> TIME
    EXECUTOR --> WEB
    EXECUTOR --> DOCS
    EXECUTOR --> CUSTOM
    
    classDef modes fill:#dbeafe,stroke:#1e3a8a,stroke-width:2px,color:#1f2937
    classDef engine fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#1f2937
    classDef tools fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    
    class COT,AGENT,AUTO,MULTI,HYBRID modes
    class PARSER,CONTEXT,EXECUTOR,SYNTHESIZER engine
    class CALC,TIME,WEB,DOCS,CUSTOM tools
```

**Features:**
- **5 Reasoning Modes**: Chain-of-Thought, Agent-Based, Auto, Multi-Step, Hybrid
- **Tool Integration**: Seamless integration with built-in and custom tools
- **Context Management**: Intelligent context handling for complex conversations
- **Response Synthesis**: High-quality response generation with confidence scoring

### **Document Processor**

The document processor provides advanced RAG capabilities with intelligent document handling.

**Processing Pipeline:**
1. **Document Upload**: Multi-format file acceptance (PDF, TXT, Images, Word)
2. **Text Extraction**: OCR for images, parsing for documents
3. **Intelligent Chunking**: Semantic-aware text splitting (1000 chars with 200 overlap)
4. **Vector Embeddings**: Generation using local models
5. **Storage**: ChromaDB vector database storage
6. **Retrieval**: Semantic search with context assembly

**Features:**
- **Multi-Format Support**: PDF, TXT, Images, Word documents, Markdown
- **Intelligent Chunking**: Semantic-aware text splitting
- **Advanced RAG**: Multi-stage retrieval with re-ranking
- **Vector Storage**: ChromaDB integration for efficient search

### **Async Ollama Client**

The async Ollama client provides high-performance, non-blocking communication with local LLMs.

**Features:**
- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient resource management
- **Request Throttling**: Rate limiting for stability
- **Response Caching**: Performance optimization
- **Streaming Support**: Real-time response streaming

### **Tool Registry**

The tool registry manages the extensible plugin architecture for BasicChat.

**Built-in Tools:**
- **Enhanced Calculator**: Advanced mathematical operations with step-by-step reasoning
- **Time Tools**: Timezone-aware time calculations and conversions
- **Web Search**: Real-time information retrieval via DuckDuckGo
- **Audio Tools**: Text-to-speech and speech-to-text capabilities

**Plugin Architecture:**
- **Tool Registration**: Automatic discovery and registration
- **Validation**: Safety and compatibility checking
- **Execution**: Secure tool execution with error handling
- **Extensibility**: Custom tool creation and integration

---

## 🔄 Data Flow

### **Standard Query Flow**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb', 'edgeLabelBackground': '#f9fafb'}}}%%
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant RE as Reasoning Engine
    participant LLM as LLM Service
    participant CACHE as Cache
    participant OLLAMA as Ollama
    
    U->>UI: Submit Query
    UI->>RE: Process Query
    RE->>CACHE: Check Cache
    
    alt Cache Hit
        CACHE-->>RE: Return Cached Response
        RE-->>UI: Display Response
        UI-->>U: Show Result
    else Cache Miss
        RE->>LLM: Generate Response
        LLM->>OLLAMA: API Call
        OLLAMA-->>LLM: Model Response
        LLM-->>RE: Processed Response
        RE->>CACHE: Store Response
        RE-->>UI: Display Response
        UI-->>U: Show Result
    end
```

### **Document Analysis Flow**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb', 'edgeLabelBackground': '#f9fafb'}}}%%
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant DP as Document Processor
    participant VISION as Vision Service
    participant EMB as Embedding Service
    participant VDB as Vector Database
    participant RE as Reasoning Engine
    
    U->>UI: Upload Document
    UI->>DP: Process Document
    
    alt Image Document
        DP->>VISION: Extract Text
        VISION-->>DP: OCR Result
    else Text Document
        DP->>DP: Parse Content
    end
    
    DP->>EMB: Generate Embeddings
    EMB-->>DP: Vector Embeddings
    DP->>VDB: Store Vectors
    VDB-->>DP: Storage Confirmation
    DP-->>UI: Processing Complete
    UI-->>U: Document Ready
    
    U->>UI: Query Document
    UI->>RE: Process Query
    RE->>VDB: Semantic Search
    VDB-->>RE: Relevant Context
    RE->>RE: Generate Response
    RE-->>UI: Display Response
    UI-->>U: Show Result
```

---

## ⚡ Performance Architecture

### **Caching Strategy**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb'}}}%%
graph TB
    subgraph "Cache Layers"
        L1[Memory Cache]
        L2[Redis Cache]
        L3[Disk Cache]
    end
    
    subgraph "Cache Types"
        RESPONSE[Response Cache]
        EMBEDDING[Embedding Cache]
        TOOL[Tool Cache]
        SESSION[Session Cache]
    end
    
    L1 --> RESPONSE
    L2 --> EMBEDDING
    L2 --> TOOL
    L3 --> SESSION
    
    classDef layers fill:#dbeafe,stroke:#1e3a8a,stroke-width:2px,color:#1f2937
    classDef types fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#1f2937
    
    class L1,L2,L3 layers
    class RESPONSE,EMBEDDING,TOOL,SESSION types
```

**Performance Features:**
- **Multi-Layer Caching**: L1 (Memory), L2 (Redis), L3 (Disk)
- **Smart Invalidation**: Context-aware cache management
- **Connection Pooling**: Efficient resource utilization
- **Async Processing**: Non-blocking operations
- **Response Streaming**: Real-time output generation

### **Async Architecture**

The system uses non-blocking asynchronous processing for responsive user experience.

**Key Features:**
- **Async Components**: LLM client, embedding, tools, cache operations
- **Concurrency Management**: Thread pools, async queues, semaphores
- **Performance Monitoring**: Real-time metrics, profiling, alerts
- **Error Recovery**: Automatic retry and fallback mechanisms

---

## 🔒 Security & Privacy

### **Privacy Architecture**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb'}}}%%
graph TB
    subgraph "Privacy Features"
        LOCAL[Local Processing]
        NO_TELEMETRY[No Telemetry]
        ENCRYPTED[Encrypted Storage]
        ISOLATION[Data Isolation]
    end
    
    subgraph "Security Measures"
        VALIDATION[Input Validation]
        SANITIZATION[Data Sanitization]
        ACCESS[Access Control]
        AUDIT[Audit Logging]
    end
    
    LOCAL --> VALIDATION
    NO_TELEMETRY --> SANITIZATION
    ENCRYPTED --> ACCESS
    ISOLATION --> AUDIT
    
    classDef privacy fill:#dbeafe,stroke:#1e3a8a,stroke-width:2px,color:#1f2937
    classDef security fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#1f2937
    
    class LOCAL,NO_TELEMETRY,ENCRYPTED,ISOLATION privacy
    class VALIDATION,SANITIZATION,ACCESS,AUDIT security
```

**Security Features:**
- **Local-Only Processing**: No data leaves the user's system
- **No Telemetry**: Complete privacy with no tracking
- **Encrypted Storage**: All data encrypted at rest
- **Input Validation**: Comprehensive input sanitization
- **Access Control**: Role-based access management
- **Audit Logging**: Complete audit trail for compliance

### **Data Flow Security**

Secure data handling throughout the processing pipeline ensures complete privacy and security.

**Security Measures:**
- **Input Validation**: Comprehensive input sanitization and validation
- **Process Isolation**: Secure processing with memory protection
- **Encrypted Storage**: All data encrypted at rest and in transit
- **Access Control**: Role-based permissions and authentication
- **Audit Trail**: Complete logging of all data operations

---

## 📈 Scalability Considerations

### **Horizontal Scaling**

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif', 'primaryColor': '#1e3a8a', 'primaryTextColor': '#1f2937', 'primaryBorderColor': '#374151', 'lineColor': '#6b7280', 'secondaryColor': '#f3f4f6', 'tertiaryColor': '#e5e7eb'}}}%%
graph TB
    subgraph "Load Balancing"
        LB[Load Balancer]
        HEALTH[Health Checks]
        SCALING[Auto Scaling]
    end
    
    subgraph "Service Instances"
        INSTANCE1[Instance 1]
        INSTANCE2[Instance 2]
        INSTANCE3[Instance 3]
    end
    
    subgraph "Shared Resources"
        DB[Shared Database]
        CACHE[Shared Cache]
        STORAGE[Shared Storage]
    end
    
    LB --> HEALTH
    HEALTH --> SCALING
    
    SCALING --> INSTANCE1
    SCALING --> INSTANCE2
    SCALING --> INSTANCE3
    
    INSTANCE1 --> DB
    INSTANCE2 --> CACHE
    INSTANCE3 --> STORAGE
    
    classDef loadbalancing fill:#dbeafe,stroke:#1e3a8a,stroke-width:2px,color:#1f2937
    classDef instances fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#1f2937
    classDef resources fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    
    class LB,HEALTH,SCALING loadbalancing
    class INSTANCE1,INSTANCE2,INSTANCE3 instances
    class DB,CACHE,STORAGE resources
```

**Scaling Features:**
- **Microservices Architecture**: Independent service scaling
- **Load Balancing**: Intelligent traffic distribution
- **Auto Scaling**: Automatic resource management
- **Shared Resources**: Efficient resource utilization
- **Message Queues**: Asynchronous processing

### **Vertical Scaling**

The system supports vertical scaling through resource optimization and performance tuning.

**Optimization Areas:**
- **Resource Optimization**: CPU, memory, GPU, storage tuning
- **Performance Tuning**: Cache, database, network, application optimization
- **Monitoring**: Resource monitoring, bottleneck detection, optimization recommendations

---

## 🛠️ Technology Stack

### **Core Technologies**

BasicChat is built on modern, scalable technologies:

**Frontend:**
- **Streamlit**: Primary web interface
- **React Components**: Interactive UI elements
- **Custom CSS/JS**: Styling and functionality

**Backend:**
- **Python 3.12**: Core runtime environment
- **AsyncIO**: Asynchronous processing
- **FastAPI**: High-performance web framework
- **Pydantic**: Data validation and serialization

**AI/ML:**
- **LangChain**: AI orchestration framework
- **Ollama**: Local LLM inference engine
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embedding models

**Infrastructure:**
- **Redis**: High-performance caching
- **Docker**: Containerization
- **Kubernetes**: Orchestration (optional)
- **Prometheus/Grafana**: Monitoring

### **Key Libraries**

| **Category** | **Library** | **Version** | **Purpose** |
|:---|:---|:---:|:---|
| **Web Framework** | Streamlit | ≥1.28.0 | User interface |
| **AI Framework** | LangChain | ≥0.1.0 | AI orchestration |
| **Vector Database** | ChromaDB | 1.0.13 | Semantic search |
| **LLM Engine** | Ollama | Latest | Local LLM inference |
| **Caching** | Redis | ≥4.5.0 | High-performance cache |
| **Async Processing** | AsyncIO | Built-in | Asynchronous operations |
| **Data Validation** | Pydantic | ≥2.0.0 | Data validation |
| **Embeddings** | Sentence Transformers | ≥2.2.0 | Text embeddings |

---

## 📚 References

1. **Mermaid Documentation**: Knut Sveidqvist et al. *Mermaid: Markdown-inspired diagramming and charting tool*. GitHub, 2024. Available: https://mermaid.js.org/

2. **System Architecture Patterns**: Fowler, Martin. *Patterns of Enterprise Application Architecture*. Addison-Wesley, 2002.

3. **Microservices Architecture**: Newman, Sam. *Building Microservices: Designing Fine-Grained Systems*. O'Reilly Media, 2021.

4. **AI System Design**: Hulten, Geoff. *Building Intelligent Systems: A Guide to Machine Learning Engineering*. Apress, 2018.

5. **Privacy by Design**: Cavoukian, Ann. *Privacy by Design: The 7 Foundational Principles*. Information and Privacy Commissioner of Ontario, 2009.

6. **Performance Engineering**: Gregg, Brendan. *Systems Performance: Enterprise and the Cloud*. Prentice Hall, 2013.

---

*This architecture document provides a comprehensive overview of BasicChat's technical design. For implementation details, see the individual component documentation and codebase.*

[← Back to README](../README.md) | [Features →](FEATURES.md) | [Development →](DEVELOPMENT.md) | [Roadmap →](ROADMAP.md) 