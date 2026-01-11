# Stress Predictor

Tool command line untuk memprediksi region stress dari suatu sequence. Menggunakan Transformer-based model (DNABERT-2 dan Mistral DNA Athaliana) yang sudah di fine tune menggunakan data Nicotiana Tabaccum.

## Project Structure
stress-predictor/\
│\
├── pyproject.toml\
├── stress_predictor/\
│ ├── init.py\
│ ├── cli.py\
│ ├── io_utils.py\
│ ├── model_utils.py\
│ └── predict.py\

## Setup Instruction
1. Buat folder baru (contoh: Stress Predictor).
2. Buka command prompt pada direktori folder Stress Predictor
3. Clone repository
   ```
   git clone https://github.com/venusangela/stress-predictor.git
   ```
4. Buat virtual environment pada folder Stress Predictor
   Buat virtual environment (sesuaikan .venv dengan nama virtual environment yang diinginkan)
   ```
   python -m venv .venv
   ```
   Aktivasi virtual environment yang sudah dibuat
   Untuk Linux/Mac (sesuaikan .venv dengan nama virtual environmentmu)
   ```
   source .venv/bin/activate
   ```
   Untuk Windows (sesuaikan .venv dengan nama virtual environmentmu)
   ```
   .venv\Scripts\activate
   ```
5. Pindah ke direktori stress-predictor
   ```
   cd stress-predictor
   ```
6. Install dependencies
   ```
   pip install -e .
   ```

## Run Prediction
### Input
Input dari model berupa file fasta. Untuk region stress classification, input berupa sequence DNA Nicotiana dengan panjang 1000 atau 2000. Sedangkan untuk promoter stress classification, input berupa sequence DNA Nicotiana dengan panjang 5000/6000/7000/8000/9000/10000.
### Region Stress Classification
Jalankan perintah berikut pada CLI
* ganti samples/test.fasta dengan path file pasta milikmu
* model dan tokenizer bisa pilih dnabert / mistral-athaliana namun harus sama
* ganti results dengan path folder output yang diinginkan (file output berupa json)
```
stress-predictor --rg --input samples/test.fasta --model dnabert --tokenizer dnabert rg --output folder

```
### Promoter Stress Classification
Jalankan perintah berikut pada CLI
Jalankan perintah berikut pada CLI
* ganti samples/test.fasta dengan path file pasta milikmu
* model dan tokenizer bisa pilih dnabert / mistral-athaliana namun harus sama
* ganti results dengan path folder output yang diinginkan (file output berupa json)
```
stress-predictor -- pr --input samples/test.fasta --model dnabert --tokenizer dnabert --output folder

```
### Parameter
Berikut adalah parameter yang dapat digunakan.
* Input (wajib diberikan)
* Model (wajib diberikan)
* Tokenizer (wajib diberikan)
* Force CPU (Optional)
* Prediction Mode (Wajib)
    * PR
      * Slice
      * Stride
      * Output
    * RG
      * Output
* Output
Jika ingin mengetahui lebih jelas parameter yang dapat digunakan, gunakan command berikut:
```
stress-predictor --help
stress-predictor pr --help
stress-predictor rg --help
```
