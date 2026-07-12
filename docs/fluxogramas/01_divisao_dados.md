```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 38
    rankSpacing: 42
---
flowchart TB
    A([Início]) --> B[/Espectros NIR brutos<br/>&amp; Qualidade Sensorial/]
    B --> C[Padronizar espectros]

    subgraph KS["Algoritmo de Kennard-Stone"]
        direction TB
        D[Calcular distâncias<br/>euclidianas]
        E[Selecionar o par inicial<br/>mais distante]
        F[Adicionar amostra com<br/>maior distância mínima]
        G{Atingiu 20%<br/>da classe?}

        D --> E
        E --> F
        F --> G
        G -- Não --> F
    end

    C --> D
    G -- "Não selecionadas" --> H[/Treinamento da classe<br/>(80%)/]
    G -- Selecionadas --> I[/Validação da classe<br/>(20%)/]
    H --> J([Fim])
    I --> J

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238
    classDef decisao fill:#F4FAFF,stroke:#7BA8D8,color:#263238

    class A,J terminador
    class B,H,I dados
    class C,D,E,F processo
    class G decisao

    style KS fill:#F7F7F7,stroke:#D6D6D6,color:#263238
    linkStyle default stroke:#7AA695,stroke-width:2px
```
