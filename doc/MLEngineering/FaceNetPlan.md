# Face Recognition Pipeline

Input Image / Frame  
↓  
Face Detection + Alignment (InsightFace)  
↓  
Face Embedding Extraction (InsightFace)  
↓  
Embedding Normalization  
↓  
Similarity Search (Database / Index)  
↓  
Identity Decision (Thresholding)  
↓  
Result (Person ID or Unknown)
