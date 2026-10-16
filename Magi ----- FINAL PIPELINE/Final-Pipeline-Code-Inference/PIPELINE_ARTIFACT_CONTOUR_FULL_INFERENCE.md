# Artifact + contour full inference

```mermaid
flowchart TD
    A[One to three ordered page images] --> B[Local Magi detections OCR and original links]
    B --> C[Paddle OCR per original text box]
    C --> D[Local Gemma artifact classification]
    D --> E[Remove artifacts and apply punctuation fallback]
    B --> F[Local YOLO bubble detection]
    F --> G[Punctuation fallback bubble mask check]
    E --> G
    G --> H[Surviving text boxes]
    H --> I[Individual outlines faint retry and overlap groups]
    I --> J[Concatenate surviving members in reading order]
    J --> K[Contour SPOKEN or NARRATED]
    B --> L[Explicit text to tail links]
    L --> M[Tail forces SPOKEN]
    K --> M
    B --> N[Original character crops]
    N --> O[Fine tuned final two crop encoder blocks]
    O --> P[Character cleanup and cross page entities]
    P --> Q[Route groups through original member associations]
    M --> Q
    Q --> R[Raw transcript]
    R --> S[Optional local Gemma OCR correction]
    S --> T[Final transcript annotations and audits]
    U[Own stage caches only] -.-> B
    U -.-> D
    U -.-> F
```

Only the notebook's own stage caches are reused. Dependencies and hashes are exported. No labels, learned narration classifier, other notebook execution or legacy cache discovery is required.
