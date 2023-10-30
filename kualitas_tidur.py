import pickle
import streamlit as st

model = pickle.load(open('kualitas_tidur.sav','rb'))

st.title('Menilai kualitas tidur sesuai dengan gaya hidup')

Person_ID = st.number_input('Masukkan ID 1-374', 0, 100)
Age = st.number_input('Masukkan Umur', 0, 100)
Occupation = st.selectbox('Pilih Pekerjaan', ['Nurse', 'Doctor', 'Engineer', 'Lawyer', 'Teacher', ' Accountant', 'Salesperson', 'Scientist', 'Software Engineer', 'Sales Representative', 'Manager'])
Sleep_Duration = st.slider('Durasi Tidur perhari (jam)', 1, 24)
Gender = st.selectbox('Pilih jenis kelamin', ['Male', 'Female'])
Physical_Activity_Level	= st.number_input('Tingkat aktivitas fisik (menit perhari)', 0)
Stress_Level = st.slider('Tingkat stress (Stress ringan-stress berat)', 1, 10)
BMI_Category = st.selectbox('Pilih BMI Category', ['Underweight', 'Normal', 'Overweight'])
Blood_Pressure = st.selectbox('Pilih tekanan darah', ['Systolic', 'Diastolic'])
Heart_Rate = st.number_input('Detak jantung (detak permenit)', 0)
Daily_Steps = st.number_input('Jumlah langkah harian', 0)
Sleep_Disorder = st.selectbox('Pilih gangguan tidur', ['None', 'Insomnia', 'Sleep Apnia'])

predict = ''

if st.button('Estimasi Nilai kualitas tidur'):
    predict = model.predict(
        [[Person_ID, Age, Sleep_Duration, Physical_Activity_Level, Stress_Level, Heart_Rate, Daily_Steps]]
    )
    st.write('Nilai kualitas tidur :', int(predict[0]))
