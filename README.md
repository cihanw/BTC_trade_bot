# BTC Trade Bot

Python tabanli bir veri toplama ve birlestirme projesi. Binance Futures verileri ve ek piyasa metrikleri toplanip tek bir islenmis veri setinde birlestirildi.

## Son Durum

- Binance **30m Kline** verileri toplandi.
- Binance **Funding Rate** verileri toplandi.
- **Open Interest** ve **Long/Short Ratio** verileri eklendi.
- **CME BTC Futures** verileri projeye dahil edildi.
- Tum kaynaklar birlestirilerek `data/processed/` altinda merged ciktilar olusturuldu.

## Proje Yapisi

- `klines_fundingRate_30min.py`: Kline ve funding rate indirme islemleri
- `openInterest.py`: Open interest veri islemleri
- `CME.py`: CME veri cekme ve kaydetme
- `preprocess1.py`, `preprocess2.py`, `preprocess3.py`: veri temizleme ve merge adimlari
- `data/raw/`: ham veri dosyalari
- `data/processed/`: birlestirilmis veri ciktilari

## Calistirma

1. Bagimliliklari kurun:
   ```bash
   pip install pandas requests
   ```
2. Veri toplama scriptlerini calistirin:
   ```bash
   python klines_fundingRate_30min.py
   python openInterest.py
   python CME.py
   ```
3. On-isleme ve birlestirme adimlarini calistirin:
   ```bash
   python preprocess1.py
   python preprocess2.py
   python preprocess3.py
   ```
