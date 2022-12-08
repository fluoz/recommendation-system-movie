# Laporan Proyek Machine Learning - Hasan Abdullah Munshi

## Project Overview

Saat ini, rekomendasi film dari sistem berbasis server telah mempermudah pencarian film. Rekomendasi film membantu kita menemukan film yang perlu kita tonton, alih-alih mencari secara online dan membantu bioskop dan penggemar film dengan menyarankan film tingkat atas untuk ditonton tanpa melihat ke database besar yang sangat memakan waktu. Sebagai pendekatan untuk dilema ini, kami Memperkenalkan model berdasarkan pendekatan kolaboratif dan berbasis konten yang akan menggunakan berbagai algoritme Pembelajaran Mesin berbasis Python dari kumpulan data yang sangat besar dan menghasilkan saran film berdasarkan selera dan riwayat tontonan atau genre sebelumnya. Ini dibandingkan dengan sistem rekomendasi lain yang berbeda dan didasarkan pada pendekatan berbasis konten.

Tujuan utama dari sistem rekomendasi film adalah untuk memfilter dan memprediksi hanya film-film yang kemungkinan besar ingin ditonton oleh pengguna terkait. Algoritme ML untuk sistem rekomendasi ini menggunakan data tentang pengguna ini dari database sistem. Data ini digunakan untuk memprediksi film dari genre yang mirip.

## Business Understanding


### Problem Statements

Berdasarkan pada project overview di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah bagaimana mendapatkan film yang mirip seperti film yang sebelumnya pengguna tonton? 

### Goals

Tujuan proyek ini dibuat adalah untuk mempermudah pengguna dalam menemukan film yang diinginkan.

### Solution statements
Solusi yang dapat dilakukan agar goals terpenuhi adalah membuat model sistem rekomendasi untuk memberikan beberapa rekomendasi film dengan menggunakan metode _Content Based Filtering_.

## Data Understanding
Data atau dataset yang digunakan pada proyek ini diambil dari website _Kaggle_. yaitu [Top 10000 Popular Movies Dataset
](https://www.kaggle.com/datasets/omkarborikar/top-10000-popular-movies) Ini adalah kumpulan data untuk 10000 film Populer berdasarkan peringkat TMDB.

Variabel-variabel pada Top 10000 Popular Movies Dataset adalah sebagai berikut:
- _id_ : merupakan unique id dari sebuah film.
- _original_language_ : Ada total 44 bahasa yang ada di kolom ini. Total 7771 film dengan 'Bahasa Inggris' sebagai bahasa aslinya.
- _original_title_ : merupakan judul dari film tersebut.
- _popularity_ : merupakan popularitas film. Semakin besar angkanya, semakin tinggi popularitasnya.
- _release_date_ : merupakan tanggal rilis film.
- _vote_average_ : merupakan rata-rata rating/suara untuk film tersebut.
- _vote_count_ : merupakan jumlah peringkat/suara yang direkam untuk film tersebut.
- _genre_ : merupakan list genre dari sebuah film.
- _overview_ : merupakan deskripsi singkat film dalam format _string_.
- _revenue_ :  merupakan pendapatan dari sebuah film.
- _runtime_ :  merupakan runtime film dalam hitungan menit.
- _tagline_ : merupakan tagline dari sebuah film.


### Exploratory data analysis
| #  | Column            | Non-Null Count | Dtype   |
|----|-------------------|----------------|---------|
| 0  | Unnamed: 0        | 10000 non-null | int64   |
| 1  | id                | 10000 non-null | int64   |
| 2  | original_language | 10000 non-null | object  |
| 3  | original_title    | 10000 non-null | object  |
| 4  | popularity        | 10000 non-null | float64 |
| 5  | release_date      | 9962 non-null  | object  |
| 6  | vote_average      | 10000 non-null | float64 |
| 7  | vote_count        | 10000 non-null | int64   |
| 8  | genre             | 10000 non-null | object  |
| 9  | overview          | 9900 non-null  | object  |
| 10 | revenue           | 10000 non-null | int64   |
| 11 | runtime           | 9991 non-null  | float64 |
| 12 | tagline           | 7080 non-null  | object  |

Berdasarkan table di atas, kita dapat mengetahui bahwa dataset memiliki 10000 entri. disini terlihat bahwa '_release_date', '_overview_', '_runtime_', dan '_tagline_' terdapat _missing value_. namun karena kita tidak memerlukan fitur itu, maka kita hanya hapus kolomnya saja.

## Data Preparation
tahap ini melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan.
- Menghapus fitur yang tidak relevan : Pada tahap ini kita akan menghapus beberapa fitur yang tidak memberikan informasi terkait film seperti, '_original_language_', '_popularity_', '_release_date_', '_vote_average_', '_vote_count_', '_overview_', '_revenue_', '_runtime_', '_tagline_'.
- Menangani _missing value_ : Karena ada data di kolom '_genre_' yang kosong, maka diputuskan untuk mengambil semua data film yang data kolom '_genre_' nya tidak kosong.
- Mengurutan data : mengurutkan data sesuai dengan '_id_' secara ascending.
- Menangani duplikat data : menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, saya membuang data duplikat pada kolom ‘id’.
- Mengkonversi data menjadi bentuk list : kita perlu melakukan konversi data series menjadi list. Dalam hal ini, kita menggunakan fungsi tolist() dari library numpy.
- Membuat _dictionary_ : kita akan membuat dictionary untuk menentukan pasangan key-value pada data '_id_', '_movie_name_', dan '_genre_'.
- Melakukan perhitungan _idf_ : untuk menemukan representasi fitur penting dari setiap kategori '_genre_'.
- Mengubah vektor _tf-idf_ dalam bentuk matriks : Untuk menghasilkan vektor tf-idf dalam bentuk matriks.

## Modeling and Result
### Content Based Filtering
Ide dari sistem rekomendasi berbasis konten (content-based filtering) adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Misal, jika Anda menyukai film Ada Apa dengan Cinta, sistem akan merekomendasikan film dengan aktor utama Nicholas Saputra atau Dian Sastrowardoyo. Sistem juga akan merekomendasikan film dengan genre drama lainnya.


##### TF-IDF Vectorizer
TfidfVectorizer akan melakukan proses tokenisasi pada teks, mempelajari kosa kata, melakukan pembobotan frekuensi dokumen secara terbalik (inverse), dan memungkinkan Anda untuk melakukan proses encoding teks baru.


##### Cosine Similarity
Cosine Similarity mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks.

Kelebihan dan kekurangan _Content Based Filtering_ antara lain :
###### kelebihan
- Memiliki kemampuan untuk merekomendasikan item.
- Tidak memerlukan data pengguna.

###### kekurangan
- Terbatasnya rekomendasi hanya pada item-item yang mirip sehingga tidak ada kesempatan untuk mendapatkan item yang tidak terduga.
- Membutuhkan banyak pengetahuan suatu domain

##### Recommendation Result
untuk mendapatkan rekomendasi, saya menggunakan judul film 'Star Wars' yang bergenre '_Adventure_', '_Action_', '_Science Fiction_'.

|              movie_name             |                    genre                   |
|:-----------------------------------:|:------------------------------------------:|
|         ゴジラvsキングギドラ        | ['Action', 'Adventure', 'Science Fiction'] |
|               Ant-Man               | ['Science Fiction', 'Action', 'Adventure'] |
| Captain America: The Winter Soldier | ['Action', 'Adventure', 'Science Fiction'] |
|         Ant-Man and the Wasp        | ['Action', 'Adventure', 'Science Fiction'] |
| The Hunger Games: Catching Fire     | ['Adventure', 'Action', 'Science Fiction'] |

Tabel 1. Hasil Rekomendasi film 'Star Wars'

dapat dilihat bahwa genre yang dihasilkan dari rekomendasi sama persis dengan genre dari film "Star Wars". Artinya prediksinya memiliki akurasi yang baik.
## Evaluation
Metrik yang akan kita gunakan pada prediksi ini adalah _Precision_ dan _Recall_ yaitu metrik biner yang digunakan untuk mengevaluasi model dengan keluaran biner. Jadi kita membutuhkan cara untuk menerjemahkan masalah numerik kita (peringkat biasanya dari 1 sampai 5) menjadi masalah biner (item yang relevan dan tidak relevan). lalu kalau _Precision at k_ adalah proporsi item yang direkomendasikan dalam set top-k yang relevan. Interpretasinya adalah sebagai berikut. Misalkan presisi saya pada 10 dalam masalah rekomendasi 10 teratas adalah 80%. Ini berarti 80% dari rekomendasi yang dibuat relevan dengan pengguna.

_Precision at k_ didefinisikan dalam persamaan berikut

$$Precision\ at\ k = { \Large \ttvar{#}\ of\ recommended\ item\ that\ are\  relevan \over  \ttvar{#}\  of \ recommended \ item\Large}$$ 

Untuk testing disini menggunakan nama film 'Star Wars' dengan genre '_Adventure_', '_Action_', '_Science Fiction_'. Tentu kita berharap rekomendasi yang diberikan adalah film dengan kategori yang mirip. 

Pada _Tabel 1_  dapat disimpulkan bahwa dari 5 item rekomendasi semuanya mirip dengan genre dari film '_Star Wars_' karena genre yang dihasilkan sama persis yaitu '_Adventure_', '_Action_', '_Science Fiction_', jadi item yang relevan ada sebanyak 5 item, maka hasil dari precision adalah 5/5 atau 100%.

## Referensi
Piyush Kumar, Shaik Golam Kibriya, Yuva Ajay and Ilampiray (2021), Movie Recommender System Using Machine Learning Algorithms - IOPscience https://iopscience.iop.org/article/10.1088/1742-6596/1916/1/012052
