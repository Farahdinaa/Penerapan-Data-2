# Proyek Akhir: Menyelesaikan Students' Performance

## Business Understanding
Jaya Jaya Institut adalah sebuah lembaga pendidikan tinggi yang telah beroperasi sejak tahun 2000 dan telah menghasilkan lulusan berkualitas yang berkontribusi di berbagai industri. Namun, dalam beberapa tahun terakhir, mereka menghadapi tantangan serius berupa tingginya tingkat dropout mahasiswa. Fenomena ini tidak hanya berdampak pada reputasi institusi, tetapi juga mempengaruhi stabilitas keuangan dan efektivitas program akademik. Salah satu faktor utama yang menyebabkan dropout adalah kesulitan akademik, masalah finansial, serta kurangnya keterlibatan mahasiswa dalam kegiatan akademik. Jika dropout tidak segera diidentifikasi dan dicegah, institusi akan kehilangan lebih banyak mahasiswa potensial, yang pada akhirnya dapat menurunkan tingkat kelulusan dan daya saing di dunia pendidikan. Untuk mengatasi permasalahan ini, Jaya Jaya Institut berupaya menerapkan pendekatan berbasis data dengan menggunakan model machine learning untuk memprediksi mahasiswa yang berisiko tinggi mengalami dropout. Dengan sistem prediksi ini, institusi dapat memberikan intervensi lebih awal, seperti bimbingan akademik, dukungan finansial, atau konseling untuk membantu mahasiswa tetap berada di jalur pendidikan mereka. Selain itu, dashboard analisis juga akan dikembangkan agar pihak administrasi dapat dengan mudah memantau dan memahami tren performa mahasiswa serta mengambil langkah strategis untuk meningkatkan retensi mahasiswa. Proyek ini bertujuan untuk mengembangkan sistem prediksi dropout yang akurat serta dashboard interaktif yang dapat membantu pihak manajemen dalam membuat keputusan berbasis data. Dengan solusi ini, diharapkan tingkat dropout dapat ditekan, sehingga lebih banyak mahasiswa dapat menyelesaikan pendidikan mereka dengan sukses.

### Permasalahan Bisnis
1. Berapa jumlah total mahasiswa yang terdaftar, dan bagaimana tingkat kelulusan mereka?
2. Bagaimana hubungan antara pendidikan sebelumnya dengan status mahasiswa (Graduate, Dropout, Enrolled)?
3. Sejauh mana biaya kuliah mempengaruhi status mahasiswa?
4. Bagaimana tingkat kehadiran mahasiswa di semester awal berkorelasi dengan status akademik mereka?
5. Bagaimana pengaruh beasiswa terhadap kemungkinan mahasiswa lulus, dropout, atau tetap terdaftar?
6. Bagaimana hubungan antara jumlah mata kuliah yang diambil di semester awal dengan nilai masuk mahasiswa?
7. Faktor apa saja yang paling mempengaruhi mahasiswa untuk bertahan hingga lulus dibandingkan dengan mereka yang dropout?

### Cakupan Proyek
1. Pengumpulan dan Pemahaman Data
   - Dataset tersedia dalam file data.csv.
   - Memahami struktur dan karakteristik data, termasuk tipe variabel, jumlah data, serta nilai yang hilang atau tidak valid.
2. Eksplorasi Data (Exploratory Data Analysis - EDA)
   - Melakukan analisis awal untuk mengidentifikasi pola, tren, dan distribusi data.
   - Visualisasi data menggunakan Seaborn, Matplotlib, dan Plotly untuk memahami hubungan antar variabel serta mendeteksi anomali atau outlier.
   - Membuat berbagai plot seperti histogram, countplot, scatterplot, dan boxplot untuk mendapatkan wawasan mendalam.
3. Pembersihan dan Persiapan Data
   - Menangani nilai yang hilang menggunakan SimpleImputer dengan strategi most_frequent.
   - Menghapus duplikasi data untuk memastikan keakuratan analisis.
   - Menggunakan Label Encoding untuk variabel kategorikal dan StandardScaler untuk menormalisasi data numerik.
   - Menyimpan data yang telah diproses ke dalam file data_clean.csv dan data_processed.csv.
4. Penanganan Ketidakseimbangan Data
   - Menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan jumlah data kelas minoritas.
   - Menggunakan Tomek Links untuk menghilangkan contoh yang terlalu mirip dan meningkatkan separasi antar kelas.
   - Mengecek distribusi data sebelum dan sesudah balancing untuk memastikan efektivitas metode yang digunakan.
5. Pemilihan dan Pelatihan Model
   - Menerapkan berbagai algoritma klasifikasi:
     - Random Forest
     - Gradient Boosting
     - AdaBoost
     - SVM (Support Vector Machine)
     - SGDClassifier
   - Menggunakan Pipeline untuk mengotomatisasi preprocessing dan pelatihan model.
   - Melakukan Hyperparameter Tuning dengan GridSearchCV untuk mendapatkan performa optimal.
6. Evaluasi Model
   - Menggunakan Confusion Matrix, Accuracy, Precision, Recall, dan F1-Score untuk menilai kinerja model.
   - Melakukan Cross-Validation untuk menghindari overfitting dan meningkatkan generalisasi model.
   - Menyimpan model terbaik ke dalam file models/best_model.pkl.
7.Menyimpan model
   - Menyimpan model yang telah dilatih dalam format .pkl menggunakan Joblib.
   - Menyimpan preprocessor (scaler.pkl) dan encoder (encoder.pkl) untuk memastikan kompatibilitas dengan input baru.
   - Menyiapkan direktori models/ untuk menyimpan semua model yang telah dilatih.
8. Pembuatan Dashboard Interaktif
   - Menggunakan Looker Studio untuk menampilkan visualisasi performa model dan insight dari data.
   - Menghubungkan hasil prediksi dengan laporan yang dapat diakses oleh tim HR atau akademik.
     
Cakupan proyek ini dirancang untuk memberikan pemahaman mendalam tentang faktor-faktor yang berkontribusi terhadap dropout mahasiswa, serta menyediakan solusi berbasis data untuk meningkatkan tingkat kelulusan.

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

Setup Environment - Anaconda
```
conda create --name main-ds python=3.9  
conda activate main-ds  
pip install -r requirements.txt  
```
Setup Environment - Shell/Terminal
```
mkdir proyek_analisis_data  
cd proyek_analisis_data  
pipenv install  
pipenv shell  
pip install -r requirements.txt  
 
```

## Business Dashboard
Business dashboard yang telah dibuat berfungsi sebagai alat pemantauan kinerja akademik mahasiswa di Jaya Jaya Institut, dengan fokus pada tingkat kelulusan, dropout, dan faktor-faktor yang memengaruhi keberhasilan studi. Dashboard ini menyajikan visualisasi data yang mencakup jumlah total mahasiswa, tingkat kelulusan, serta analisis terhadap faktor seperti biaya kuliah, pendidikan sebelumnya, beasiswa, dan kehadiran di semester awal. Selain itu, terdapat analisis hubungan antara jumlah mata kuliah yang diambil dengan nilai masuk mahasiswa. Dengan tampilan yang interaktif dan informatif, dashboard ini membantu pihak akademik dalam mengidentifikasi pola dropout serta mengambil langkah strategis untuk meningkatkan retensi dan keberhasilan mahasiswa. Jika tersedia, dashboard ini dapat diakses melalui link berikut: https://lookerstudio.google.com/reporting/39a8d8b5-e939-4393-8ea9-265bac9e0ccd

![Untitled_Report (6)_page-0001](https://github.com/user-attachments/assets/7de93ead-37e5-4700-9e76-a2678f10ff3b)


## Menjalankan Sistem Machine Learning
1. Persiapkan Lingkungan

   Pastikan semua dependensi yang dibutuhkan sudah terinstal.
```
streamlit run Prediksi.py
```
2. Pastikan File yang Dibutuhkan Tersedia
   Letakkan file berikut di direktori proyek:
   - best_model.pkl (model machine learning yang sudah dilatih)
   - scaler.pkl (untuk normalisasi fitur numerik)
   - encoder.pkl (untuk encoding fitur kategorikal)
   - data_clean.csv (dataset bersih untuk referensi dan visualisasi)
3. Jalankan Aplikasi Streamlit
   Buka terminal atau command prompt, lalu jalankan perintah berikut di direktori tempat file kode disimpan:
 ```
streamlit run Prediksi.py
```
Interaksi dengan Aplikasi

4. Aplikasi akan menampilkan Dashboard Prediksi Dropout Mahasiswa di browser.
   - Gunakan sidebar untuk memasukkan fitur input seperti kategori dan nilai numerik.
   - Klik tombol Predict untuk melihat hasil prediksi apakah mahasiswa akan Dropout atau Continue, beserta tingkat kepercayaannya.
   - Dashboard juga akan menampilkan distribusi status mahasiswa dalam bentuk visualisasi menggunakan Plotly.

Dengan sistem ini, pihak akademik dapat dengan mudah melakukan analisis prediktif dan mengambil tindakan lebih awal terhadap mahasiswa yang berisiko mengalami dropout.
Prototype dapat dilihat di link berikut : https://submisiondata2-tjgphrpwbarcgyjlqxdjuf.streamlit.app/

## Conclusion
Proyek ini bertujuan untuk membantu Jaya Jaya Institut dalam mendeteksi mahasiswa yang berisiko mengalami dropout menggunakan teknik machine learning. Dari analisis yang dilakukan, faktor utama yang mempengaruhi dropout mencakup biaya kuliah, jumlah mata kuliah yang diambil, kehadiran di semester awal, serta riwayat pendidikan sebelumnya. Model machine learning yang dikembangkan mampu memberikan prediksi dengan akurasi yang baik dan telah diintegrasikan ke dalam dashboard interaktif berbasis Streamlit, memungkinkan pihak akademik untuk menganalisis data serta memprediksi status mahasiswa secara real-time. Dengan sistem ini, institusi dapat melakukan intervensi lebih awal melalui bimbingan khusus, sehingga membantu mengurangi angka dropout dan meningkatkan tingkat kelulusan. Ke depannya, model ini dapat terus dikembangkan dengan dataset yang lebih besar serta penambahan fitur-fitur baru agar prediksi semakin akurat dan bermanfaat.

### Rekomendasi Action Items
1. Menyediakan Program Beasiswa atau Keringanan Biaya
   Mengingat biaya kuliah menjadi salah satu faktor dropout, institusi dapat meningkatkan akses beasiswa atau menyediakan skema pembayaran fleksibel bagi mahasiswa yang mengalami kesulitan finansial.

2. Meningkatkan Keterlibatan Akademik dan Non-Akademik
   Mendorong mahasiswa untuk lebih aktif dalam kegiatan ekstrakurikuler, mentoring, atau program tutor sebaya guna meningkatkan keterikatan mereka dengan lingkungan kampus.

3. Optimalisasi Kurikulum dan Metode Pengajaran
   Melakukan evaluasi terhadap jumlah mata kuliah dan beban studi, serta menerapkan metode pembelajaran yang lebih fleksibel seperti hybrid learning untuk membantu mahasiswa yang mengalami kesulitan akademik.

4. Meningkatkan Kehadiran di Semester Awal
   Mengembangkan program orientasi dan monitoring khusus bagi mahasiswa baru agar mereka lebih siap dalam menghadapi perkuliahan dan mengurangi kemungkinan dropout di awal studi.
