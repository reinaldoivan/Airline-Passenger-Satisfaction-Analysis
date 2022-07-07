# Airline Passenger Satisfaction Analysis
**Analysis & Machine Learning of Airline Passenger Satisfaction Database**

*Associated with Purwadhika Coding School for Final Project*

<br />

Business Problem Understanding
------------------------------
**Problem Statement :**

Faktor apa saja yang bisa mempengaruhi kepuasan pelanggan di industri penerbangan?

Sebagian besar perusahaan menyadari bahwa menyediakan pelanggan mereka dengan pengalaman yang terbaik adalah kebutuhan strategis, tetapi kebanyakan menyatakan bahwa mereka tidak dapat mengelolanya secara efektif. Masalah ini diperparah karena banyaknya poin interaksi dengan pelanggan didalam industri penerbangan.

Jika kita bisa memanfaatkan analisis data untuk mengantisipasi kepuasan pelanggan dan faktor-faktor apa saja yang bisa mempengaruhi kepuasan pelanggan, kita bisa melakukan penanggulangan sebelum adanya ketidakpuasan dari pelanggan.

**Goals :**

Dalam projek kali ini kita membahas dalam sudut pandang maskapai penerbangan dan akan menjawab beberapa pertanyaan:
1. Apa saja insight yang dapat diambil dari data yang dimiliki?
2. Faktor apa saja yang mempengaruhi penilaian pelanggan secara signifikan?
3. Apakah ada pengembangan yang dapat dilakukan untuk meningkatkan tingkat kepuasan pelanggan?

**Analytic Approach :**

Untuk menjawab pertanyaan-pertanyaan di atas, maka kita akan menganalisis data untuk menemukan hal-hal apa saja yang berpengaruh dalam penilaian kepuasan pelanggan, dimana kemudian kita akan membangun model **klasifikasi** yang akan membantu perusahaan untuk melihat perbandingan pelanggan yang puas dan tidak serta langkah apa saja yang perlu diambil untuk memperbaikinya.

**Metric Evaluation :**

Berdasarkan konsekuensinya, maka kita akan membuat model yang dapat meminimalisir 2 hal:
1. Jumlah pelanggan yang dianggap puas tetapi justru tidak *(False Positive)*, karena dapat merugikan perusahaan (pelanggan tidak kembali) dan juga dapat menyebabkan kehilangan pelanggan baru (word of mouth).
2. Jumlah insentif yang diberikan kepada pelanggan *(False Negative)* sehingga cost yang keluar lebih efisien.

Meskipun terlihat bahwa konsekuensi dari False Positive lebih besar, kita juga tetap harus memperhatikan konsekuensi yang didapat dari False Negative. Oleh karena itu, kita akan menggunakan **f1-score** sebagai 
measurement.

<br />

Data Understanding
------------------
Dataset source: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Dataset ini merupakan survei kepuasan pelanggan sebuah airline yang disusun oleh **John D pada tahun 2018** (https://www.kaggle.com/datasets/johndddddd/customer-satisfaction), yang kemudian dimodifikasi oleh **TJ Klien pada tahun 2020** untuk tujuan klasifikasi. Berdasarkan **jangka waktu tersebut**, maka data ini valid untuk digunakan dalam membantu memecahkan masalah yang ada.

Setiap baris dari dataset mewakili data penerbangan seorang pelanggan serta kepuasan yang mereka rasakan dari servis yang tersedia. (untuk detil bisa dilihat di section selanjutnya)

### Attribute Information

| Attribute | Data Type, Length | Description |
| --- | --- | --- |
| Gender | Text | Gender of the passengers (Female, Male) |
| Customer Type | Text | The customer type (Loyal customer, disloyal customer) |
| Age | Int | The actual age of the passengers |
| Type of Travel | Text | Purpose of the flight of the passengers (Personal Travel, Business Travel) |
| Class | Text | Travel class in the plane of the passengers (Business, Eco, Eco Plus) |
| Flight distance | Int |The flight distance of this journey |
| Inflight wifi service | Int | Satisfaction level of the inflight wifi service (0:Not Applicable;1-5) |
| Departure/Arrival time convenient | Int | Satisfaction level of Departure/Arrival time convenient |
| Ease of Online booking | Int | Satisfaction level of online booking |
| Gate location | Int | Satisfaction level of Gate location |
| Food and drink | Int | Satisfaction level of Food and drink |
| Online boarding | Int | Satisfaction level of online boarding |
| Seat comfort | Int | Satisfaction level of Seat comfort |
| Inflight entertainment | Int | Satisfaction level of inflight entertainment |
| On-board service | Int | Satisfaction level of On-board service |
| Leg room service | Int | Satisfaction level of Leg room service |
| Baggage handling | Int | Satisfaction level of baggage handling |
| Check-in service | Int | Satisfaction level of Check-in service |
| Inflight service | Int | Satisfaction level of inflight service |
| Cleanliness | Int | Satisfaction level of Cleanliness |
| Departure Delay in Minutes | Int | Minutes delayed when departure |
| Arrival Delay in Minutes | Float | Minutes delayed when Arrival |
| satisfaction | Int | Airline satisfaction level(Satisfied, neutral or dissatisfied) |

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Data%20Condition.PNG)

<br />

Data Analysis
-------------
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Satisfaction%20by%20Class.PNG)
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Class%20by%20Flight%20Distance.PNG)
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Customer%20Type%20by%20Age%20Group.PNG)
![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Class%20by%20Type%20of%20Travel.PNG)

**Analysis :**

- Data balanced, dimana jumlah customer yang tidak `tidak satisfied/netral [0]` sebanyak 57% dan yang `satisfied [1]` 43%
- Berdasarkan feature `Class`:
  - Customer lebih banyak mengambil kelas Business dan Eco dibanding dengan Eco Plus.
  - Jumlah pelanggan yang puas hampir berbanding terbalik antar kedua kelas, dimana terdapat lebih banyak pelanggan yang puas di kelas Business.
  - Pelanggan pada kelompok umur di bawah 30 dan di atas 60 kebanyakan mengambil kelas Eco, dimana kelompok umur 40 sampai dengan 49 tahun kebanyakan mengambil kelas Business.
- Berdasarkan feature `Flight Distance`, pelanggan dengan penerbangan yang jauh cenderung mengambil penerbangan kelas Business.
- Berdasarkan feature `Customer Type`:
  - Jumlah customer yang loyal (82%) lebih tinggi dibanding yang tidak (18%)
  - Umur pelanggan yang loyal relatif lebih tua dibanding yang tidak.
- Berdasarkan feature `Type of Travel`, pelanggan yang mengambil penerbangan kelas Business cenderung memiliki tujuan sehubungan bisnis.

*Note: Analisis yang ditampilkan hanya yang erat kaitannya dengan model dan rekomendasi. Untuk kelengkapan tabel dan analisis bisa dilihat di file notebook*

<br />

Modelling & Evaluation
----------------------
Setelah melakukan cross validation, model yang terbaik digunakan adalah `CatBoost` dengan f1-score 0.956894 sebelum di tuning, dan 0.957916 setelah di tuning. Kedua score didapat dari training set, dan mengalami perubahan di test set dimana XGBoost default memiliki f1-score lebih tinggi, comparison dapat dilihat di gambar berikut:

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Score%20Comparison.PNG)

**Feature Importances :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Feature%20Importances.PNG)

Terlihat bahwa ternyata untuk model CatBoost kita, feature/kolom `inflight wifi service` adalah yang paling penting, kemudian diikuti dengan `Type of Travel`, `Customer Type`, dan selanjutnya. Grafik ini akan kita gunakan sebagai acuan dalam memberikan rekomendasi di section berikutnya.

**SHAP Values :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/SHAP.PNG)

Berdasarkan **SHAP Values**, kita dapat melihat:
1. `Class`: Kelas Business berpengaruh secara positif terhadap target.
2. `Type of Travel`: Jenis travel Business berpengaruh secara positif terhadap target.
3. `Customer Type`: Pelanggan yang setia berpengaruh secara positif terhadap target.
4. `Total Delay`: Total waktu terlambat yang kecil berpengaruh secara positif terhadap target.
5. Secara keseluruhan, semakin tinggi nilai kepuasan masing-masing features berpengaruh positif terhadap target, tetapi ada beberapa yang justru berpengaruh negatif seperti `Gate Location` dan `Ease of Online Booking`.

<br />

Conclusion & Recommendation
---------------------------
**Confusion Matrix :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Confusion%20Matrix.PNG)

Informasi general seputar tiket:
- Rata-rata tiket per orang = 116 USD (tidak dipisah oleh `class`)
- Campaign/Incentive Expense = 6 USD (4.5%-5% dari revenue, kita mengambil kisaran terbesar)

Summary:
- Retained customer = 1.339.452 (TN)
- Campaign Cost = 69.282 (TN) + 3.066 (FN) = 72.348 USD
- Potential Loss = 45.124 USD (FP)
- Potential Save = 50.964 USD (TP)
- Total possible income after campaign cost = 1.273.034 USD

<br />

**Conclusion :**

![](https://github.com/reinaldoivan/Airline-Passenger-Satisfaction-Analysis/blob/main/Images/Classification%20Report%20CatBoost.PNG)

Hal-hal yang dapat dikonklusikan berdasarkan hasil classification report:
- Berdasarkan `Recall`, terdapat 98% pelanggan yang perlu diberikan insentif dan seharusnya tidak puas, dan terdapat 94% pelanggan yang tidak perlu diberikan insentif dan seharusnya puas.
- Berdasarkan `Precision`, kita dapat memprediksi 96% pelanggan yang tidak puas dengan tepat, dan memprediksi 97% pelanggan yang puas dengan tepat.
- Berdasarkan `Accuracy`, kita dapat memprediksi 96% pelanggan yang seharusnya puas dan tidak puas dengan tepat.
- Berdasarkan `ROC AUC`, kita dapat mampu membedakan dua kelas (satisfied dan dissatisfied/neutral) dengan hampir sempurna.

<br />

**Recommendation :**

Hal-hal yang bisa dilakukan untuk mempertahankan pelanggan:
- Memberikan insentif pada feature `inflight wifi service`, seperti: 
  - Menawarkan harga wifi yang lebih terjangkau (bisa dipaketkan di saat booking online), tetapi harga on the spot tetap sesuai untuk mengurangi cost.
  - Menawarkan 15-30 menit wifi gratis untuk setiap pelanggan kelas eco dan eco plus yang melakukan penerbangan di atas batas tertentu (3000 miles / 6 jam) atau mengalami delay di atas batas tertentu (60 menit)
- Berdasarkan EDA, terlihat bahwa `Type of Travel` paling banyak adalah business travel, yaitu sebesar 69% dari keseluruhan pelanggan. Maka, kita dapat mengembangkan campaign B2B yang sudah ada untuk memberikan upgrade dari kelas Eco/Eco Plus menjadi kelas Business dengan harga spesial, karena kelas Business cenderung memiliki tingkat kepuasan yang lebih tinggi dibanding kelas lainnya. 
- Dari EDA kita sudah bisa melihat `loyalitas` pelanggan sangat mempengaruhi pengelompokkan pelanggan, baik dari usia, class, dan type of travel. Secara umum, mileage bonus bisa membantu pelanggan untuk melakukan penerbangan lebih sering, maka kita dapat memvariasikan insentif yang mereka bisa dapat menggunakan mileage bonus, seperti penggunaan lounge di airport ataupun hotel di tempat tujuan.
- Me-review kembali dan mengembangkan sistem `Online Boarding` terutama pada kelas Eco dan Eco Plus, karena di kelas Business memiliki tingkat kepuasan lebih tinggi.
- Berdasarkan EDA, terlihat bahwa `class` business dan eco terbagi cukup rata (48:44), namun lebih banyak pelanggan yang tidak puas di kelas ekonomi (36% dari 44%) dibanding kelas business (14% dari 48%). Maka kita dapat memprioritaskan penanganan kepuasan pelanggan di kelas bisbusinessnis.
- Memberikan insentif pada feature `Age`, seperti:
  - Pelanggan yang berumur relatif lebih muda (dibawah 30) dapat difokuskan untuk menjadi pelanggan setia dengan memberikan bonus mileage lebih, melihat kelompok umur tersebut cenderung lebih sering bepergian.
  - Pelanggan yang berumur relatif lebih tua (diatas 30) dapat difokuskan untuk upgrade ke Business Class dengan memberikan harga promo sebagai Senior Discount Campaign.
- Terlihat terdapat 3 servis di dalam grup `Airport experience` yang berpengaruh cukup tinggi, maka dapat kita memfokuskan resource untuk meningkatkan kualitas service di airport.

<br />

Hal-hal yang bisa dilakukan untuk mengembangkan project dan modelnya lebih baik lagi:
- Mencoba algoritma ML dan model lainnya, kemudian di hyperparameter tuning dengan lebih baik lagi.
- Melakukan survei lebih detil di fitur yang berpengaruh secara signifikan kepada kepuasan pelangaan, seperti `Online Boarding` dan `Inflight wifi service` sehingga dapat meningkatkan performa model.

<br />

**Limitation :**

Model paling baik digunakan dengan beberapa batas nilai feature, seperti:
- `Age`: 7-85 tahun
- `Flight Distance`: 31-4.983 miles
- `Departure/Arrival Delay`: 38-1.592 minutes

