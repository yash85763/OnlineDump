flowchart TD
A[Chat Interface] –> B[User Query]
B –> C{Analyze the Query<br/>Possibly rewrite<br/><br/>Do I need additional<br/>data for the answer?}

```
C -->|Yes| D[LLM Agent]
C -->|No| E{Rewrite the Query}

D --> F[Rerank Search Results]
F --> G[MCP Server 1]
F --> H[MCP Server 2] 
F --> I[MCP Server 3]

G --> J[Generate Answer]
H --> J
I --> J

E -->|Yes| K[Rewrite the Query]
E -->|No| L{Analyze the Answer<br/><br/>Is the answer<br/>correct/relevant to<br/>the Query?}

K --> M[LLM Agent]
M --> J

L -->|Yes| N[Answer]
L -->|No| O[Back to Query Analysis]

O --> C
N --> A
J --> L

%% Styling
classDef llmStyle fill:#f9f,stroke:#333,stroke-width:2px
classDef mcpStyle fill:#bbf,stroke:#333,stroke-width:2px
classDef processStyle fill:#bfb,stroke:#333,stroke-width:2px
classDef decisionStyle fill:#fbb,stroke:#333,stroke-width:2px

class D,M llmStyle
class G,H,I mcpStyle
class A,B,F,J,K,N processStyle
class C,E,L decisionStyle
```