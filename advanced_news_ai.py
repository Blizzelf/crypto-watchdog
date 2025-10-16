# advanced_news_ai.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any

# Model adını Hugging Face'den alıyoruz.
MODEL_NAME = "ProsusAI/finbert"
# Modeli ve tokenizer'ı sadece bir kere, program başlarken yükleyeceğiz.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_headlines_with_finbert(headlines: List[str]) -> List[Dict[str, Any]]:
    """
    Verilen haber başlıklarını FinBERT modelini kullanarak analiz eder.
    Her başlık için bir duygu etiketi (pozitif, negatif, nötr) ve skor döndürür.
    """
    if not headlines:
        return []

    results = []
    try:
        # Haberleri modelin anlayacağı formata çevir (tokenize et)
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt', max_length=512)

        # Modelden tahminleri al
        with torch.no_grad(): # Hesaplama gradyanlarını takip etme, daha hızlı çalışır
            outputs = model(**inputs)
        
        # Olasılıkları hesapla (softmax fonksiyonu ile)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Etiketler: 0 -> pozitif, 1 -> negatif, 2 -> nötr
        labels = ["positive", "negative", "neutral"]
        
        # Her bir haber başlığı için sonucu işle
        for i in range(len(headlines)):
            headline_text = headlines[i]
            # En yüksek olasılığa sahip etiketi bul
            predicted_label_index = torch.argmax(predictions[i]).item()
            predicted_label = labels[predicted_label_index]
            # O etiketin olasılık skorunu al
            predicted_score = predictions[i][predicted_label_index].item()
            
            results.append({
                "headline": headline_text,
                "label": predicted_label,
                "score": predicted_score
            })
            
    except Exception as e:
        print(f"FinBERT analizi sırasında hata oluştu: {e}")
        # Hata durumunda boş liste döndürerek programın çökmesini engelle
        return []
        
    return results