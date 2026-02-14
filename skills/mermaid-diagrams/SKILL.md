---
name: mermaid-diagrams
description: Creating Mermaid diagrams for documentation. Use when users want flowcharts, sequence diagrams, class diagrams, ER diagrams, state diagrams, Gantt charts, git graphs, or any other Mermaid visualization. Provides syntax reference and best practices for embedding diagrams in markdown files.
---

# Mermaid Diagram Expert

You are an expert at creating and optimizing Mermaid diagrams embedded in markdown files.

## Core Workflow

1. **Understand the Request**: Clarify diagram type and scope if ambiguous
2. **Generate Diagram**: Write Mermaid code block directly into markdown file
3. **Iterate**: Refine based on feedback by editing the code block

## Output Format

Always output Mermaid diagrams as fenced code blocks with `mermaid` language tag:

````markdown
```mermaid
graph LR
    A[Start] --> B[End]
```
````

These render automatically in GitHub, GitLab, Confluence, Notion, and most modern documentation tools.

## Diagram Types

### Flowcharts (`graph` or `flowchart`)

Direction options: `LR` (left-right), `TB` (top-bottom), `RL`, `BT`

```mermaid
graph LR
    A[Start] --> B{Decision}
    B -->|Yes| C[Action]
    B -->|No| D[End]

    style A fill:#e1f5ff
    style C fill:#d4edda
```

Node shapes:
- `[text]` - Rectangle
- `(text)` - Rounded rectangle
- `{text}` - Diamond (decision)
- `[(text)]` - Cylinder (database)
- `((text))` - Circle
- `>text]` - Flag

Link types:
- `-->` - Arrow
- `---` - Line
- `-.->` - Dotted arrow
- `==>` - Thick arrow
- `-->|label|` - Arrow with text

### Sequence Diagrams (`sequenceDiagram`)

⚠️ **Do NOT use `style` statements** - not supported in sequence diagrams

```mermaid
sequenceDiagram
    participant U as User
    participant A as App
    participant API

    U->>A: Login
    A->>API: Authenticate
    API-->>A: Token
    A-->>U: Success

    Note over A,API: Async validation
    
    alt Success
        A->>U: Welcome
    else Failure
        A->>U: Error message
    end
```

Arrow types:
- `->>` - Solid arrow
- `-->>` - Dotted arrow
- `-x` - Cross (error)
- `-)` - Async

### Class Diagrams (`classDiagram`)

```mermaid
classDiagram
    class User {
        +String name
        +String email
        +login() bool
        -validatePassword() bool
    }
    class Order {
        +int id
        +Date created
        +getTotal() float
    }
    User "1" --> "*" Order : places
```

Relationships:
- `<|--` - Inheritance
- `*--` - Composition
- `o--` - Aggregation
- `-->` - Association
- `..>` - Dependency

Visibility:
- `+` Public
- `-` Private
- `#` Protected

### Entity Relationship (`erDiagram`)

```mermaid
erDiagram
    USER ||--o{ ORDER : places
    ORDER ||--|{ LINE_ITEM : contains
    PRODUCT ||--o{ LINE_ITEM : "appears in"

    USER {
        int id PK
        string email UK
        string name
        date created_at
    }

    ORDER {
        int id PK
        int user_id FK
        date order_date
        string status
    }
```

Cardinality:
- `||` - Exactly one
- `o|` - Zero or one
- `}|` - One or more
- `}o` - Zero or more

### State Diagrams (`stateDiagram-v2`)

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing : start
    Processing --> Complete : finish
    Processing --> Failed : error
    Complete --> [*]
    Failed --> Idle : retry

    state Processing {
        [*] --> Validating
        Validating --> Executing
        Executing --> [*]
    }
```

### Gantt Charts (`gantt`)

```mermaid
gantt
    title Project Timeline
    dateFormat YYYY-MM-DD
    
    section Planning
    Requirements    :a1, 2024-01-01, 14d
    Design          :a2, after a1, 7d
    
    section Development
    Backend         :b1, after a2, 21d
    Frontend        :b2, after a2, 21d
    
    section Testing
    QA              :c1, after b1, 14d
```

### Git Graphs (`gitGraph`)

```mermaid
gitGraph
    commit
    commit
    branch develop
    checkout develop
    commit
    commit
    checkout main
    merge develop
    commit
```

### User Journey (`journey`)

```mermaid
journey
    title User Onboarding
    section Sign Up
        Visit site: 5: User
        Fill form: 3: User
        Verify email: 2: User
    section First Use
        Complete tutorial: 4: User
        Create first item: 5: User
```

### Pie Charts (`pie`)

```mermaid
pie title Distribution
    "Category A" : 45
    "Category B" : 30
    "Category C" : 25
```

## Styling

### Theme Directives

Add at the top of your diagram:

```mermaid
%%{init: {'theme': 'forest'}}%%
graph LR
    A --> B
```

Available themes: `default`, `forest`, `dark`, `neutral`, `base`

### Node Styling (Flowcharts)

```mermaid
graph LR
    A[Node] --> B[Styled]
    
    style A fill:#e1f5ff,stroke:#0077b6,stroke-width:2px
    style B fill:#d4edda,stroke:#28a745,color:#155724
    
    classDef highlight fill:#ffd700,stroke:#333
    class A highlight
```

### Link Styling

```mermaid
graph LR
    A --> B
    linkStyle 0 stroke:#ff0000,stroke-width:2px
```

## Best Practices

### Choosing Diagram Type

| Use Case | Diagram Type |
|----------|--------------|
| Process/workflow | `graph` / `flowchart` |
| API/system interactions | `sequenceDiagram` |
| Code structure | `classDiagram` |
| Database schema | `erDiagram` |
| Lifecycle/states | `stateDiagram-v2` |
| Project timeline | `gantt` |
| Version control flow | `gitGraph` |
| User experience | `journey` |

### Layout Tips

- **Horizontal (`LR`)**: Good for linear processes, pipelines
- **Vertical (`TB`)**: Good for hierarchies, decision trees
- Use subgraphs to group related nodes
- Keep labels concise to avoid overlap

### Readability

- Use meaningful node IDs (`UserService` not `A`)
- Add participant aliases in sequence diagrams
- Use notes and comments for context
- Limit diagram complexity—split large diagrams

## Common Issues

| Problem | Solution |
|---------|----------|
| Syntax error | Check arrow syntax, quotes around special chars |
| Layout messy | Try different direction (LR vs TB) |
| Text overlap | Shorten labels or increase spacing |
| Style not working | Sequence diagrams don't support `style` |
| Special characters | Wrap text in quotes: `A["Text with (parens)"]` |

## Escaping Special Characters

```mermaid
graph LR
    A["Node with [brackets]"]
    B["Text with (parens)"]
    C["Quote: #quot;hello#quot;"]
```

## Proactive Behavior

- Generate complete, runnable Mermaid code
- Choose sensible diagram type based on context
- Use clear, descriptive node labels
- Suggest improvements when you see opportunities
- Keep diagrams focused—split complex flows into multiple diagrams
