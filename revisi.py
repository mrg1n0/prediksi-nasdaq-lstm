import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import time
from keras.optimizers import Adam
import mysql.connector
import bcrypt
from PIL import Image

# Database connection parameters
host_name = "localhost"
user_name = "root"
user_password = ""
db_name = "revisi"

# Connect to MySQL database
def create_connection():
    return mysql.connector.connect(
        host=host_name,
        user=user_name,
        password=user_password,
        database=db_name
    )

# Create users table if not exists
def create_users_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL
    )
    """)
    conn.close()

def create_result_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS result (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        stock_code VARCHAR(10) NOT NULL,
        rentan_waktu VARCHAR(20) NOT NULL,
        jumlah_data INT NOT NULL,
        kolom_prediksi VARCHAR(50) NOT NULL,
        persentase_train_size INT NOT NULL,
        epochs INT NOT NULL,
        batch_size INT NOT NULL,
        rmse FLOAT NOT NULL,
        mape FLOAT NOT NULL,
        rsquared FLOAT NOT NULL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.close()

def save_result(username, stock_code, rentan_waktu, jumlah_data, kolom_prediksi, persentase_train_size, epochs, batch_size, rmse, mape, rsquared):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO result (username, stock_code, rentan_waktu, jumlah_data, kolom_prediksi, persentase_train_size, epochs, batch_size, rmse, mape, rsquared) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (username, stock_code, rentan_waktu, jumlah_data, kolom_prediksi, persentase_train_size, epochs, batch_size, rmse, mape, rsquared)
    )
    conn.commit()
    conn.close()

# Signup function
def signup(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    conn.commit()
    conn.close()

# Signin function
def signin(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
        return True
    return False

# Function to get NASDAQ stock symbols
def get_nasdaq_symbols():
    nasdaq = pd.read_csv('https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv')
    return nasdaq.Symbol.tolist()

# Main app function
def main():
   
    # Create users and result tables
    create_users_table()
    create_result_table()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if st.session_state.logged_in:
        st.sidebar.write(f"Welcome, {st.session_state.username}")
        if st.sidebar.button('Signout'):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()  # Restart the script to update the UI
    else:
        menu = ['Signin', 'Signup']
        choice = st.sidebar.selectbox('Menu', menu)

        if choice == 'Signup':
            st.subheader('Create New Account')
            new_user = st.text_input('Username')
            new_password = st.text_input('Password', type='password')
            if st.button('Signup'):
                signup(new_user, new_password)
                st.success('You have successfully created an account')
                st.info('Go to the Signin Menu to login')

        elif choice == 'Signin':
            st.subheader('Login to Your Account')
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            if st.button('Signin'):
                if signin(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f'Welcome {username}')
                    st.experimental_rerun()  # Restart the script to update the UI
                else:
                    st.warning('Incorrect Username/Password')

    if st.session_state.logged_in:
        # Rest of your data processing and prediction code goes here

        st.title('PREDIKSI HARGA INDEX :blue[NASDAQ] dengan :red[LSTM]')
        st.caption('Model menggunakan 5 parameter :blue[close, open, low, high & volume] dengan :red[Optimasi Adam]')
         # Get NASDAQ stock symbols
        nasdaq_symbols = get_nasdaq_symbols()

        # Input symbol saham dari user
        stock_symbol = st.selectbox('Pilih simbol saham NASDAQ', nasdaq_symbols)

        try:
            # Unduh data harga indeks Nasdaq menggunakan yfinance
            years = st.slider('Pilih data dalam periode (tahun)', min_value=1, max_value=10, value=5)
            rentan_waktu = f"{datetime.now().year - years}-{datetime.now().year}"
            days = years * 365
            data = yf.download(stock_symbol, period=f'{days}d', interval='1d')

            # Data Cleaning: Menghapus data duplikat dan mengisi nilai yang hilang
            data = data.drop_duplicates().fillna(method='ffill')

             # Menampilkan info jika data berhasil diunduh
            st.success(f'Data untuk saham {stock_symbol}')

            # Tampilkan data dalam bentuk tabel
            st.dataframe(data)
            st.info(f'Jumlah Data: {len(data)}')
            
            # Kolom-kolom yang digunakan untuk analisis
            analysis_columns = ['Volume', 'Open', 'Close', 'High', 'Low']

            # Tampilkan select box untuk memilih kolom prediksi
            kolom_prediksi = st.selectbox('Pilih kolom untuk prediksi', ['Close', 'Open', 'High', 'Low', 'Volume'])

            # Ambil hanya kolom yang dipilih untuk analisis
            data_analysis = data[analysis_columns]

            # Konversi data ke bentuk numpy array
            dataset = data_analysis.values.astype('float32')

            # Scaling data menjadi rentang antara 0 dan 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)

            # Tampilkan slider untuk memilih proporsi data pelatihan
            train_proporsi = st.slider('Proporsi Data Pelatihan (%)', min_value=1, max_value=100, value=80)
            train_size = int(len(dataset) * (train_proporsi / 100))
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            persentase_train_size = (len(train) / len(dataset)) * 100

            # Tampilkan informasi tentang data pelatihan dan pengujian
            train_test_info = {
                "Data": ["Pelatihan", "Pengujian"],
                "Jumlah Data": [len(train), len(test)]
            }
            st.table(train_test_info)

        except Exception as e:
            st.error(f'Error: {e}')

        # Membuat fungsi untuk membuat data dalam bentuk array 2D (sample, timestep)
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), :]
                dataX.append(a)
                dataY.append(dataset[i + look_back, analysis_columns.index(kolom_prediksi)])
            return np.array(dataX), np.array(dataY)

        # Menentukan jumlah timestep (look_back)
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # Membuat model LSTM dengan fungsi aktivasi ReLU
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(look_back, len(analysis_columns))))
        model.add(Dense(1))

        # Compile model menggunakan optimisasi ADAM dan fungsi kerugian Mean Squared Error
        model.compile(loss='mean_squared_error', optimizer=Adam())

        # Reshape data menjadi bentuk 3D (sample, timestep, feature)
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(analysis_columns)))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], len(analysis_columns)))

        # Buat select slider untuk memilih jumlah epochs
        epochs = st.slider('Jumlah Epochs', min_value=1, max_value=100, value=100)

        # Buat select slider untuk memilih ukuran batch
        batch_size = st.slider('Ukuran Batch Size', min_value=1, max_value=100, value=1)

        # Buat select slider dengan nilai default 7 hari dan rentang 1-30 
        selected_days = st.slider("Jumlah Prediksi Hari Pada Masa Depan :", min_value=1, max_value=30, value=7)

        progress_text = "Pelatihan model sedang berlangsung. Harap tunggu..."
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)

        # Proses pelatihan model
        if st.button("Mulai Pelatihan Model"):
            
            my_bar.text(progress_text)  # Mengatur teks progres bar

            for epoch in range(epochs):
                # Fitting model to the training data
                history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0)
                loss = history.history['loss'][0]
                my_bar.progress((epoch + 1) / epochs)

            # Melakukan prediksi pada data training dan testing
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            # Inverse transform prediksi agar dapat dibandingkan dengan data asli
            scaler_single = MinMaxScaler(feature_range=(0, 1))
            scaler_single.fit_transform(data[[kolom_prediksi]].values.astype('float32'))
            
            trainPredict = scaler_single.inverse_transform(trainPredict)
            trainY_inverse = scaler_single.inverse_transform([trainY])
            testPredict = scaler_single.inverse_transform(testPredict)
            testY_inverse = scaler_single.inverse_transform([testY])

            # Menambahkan kolom tanggal
            dates = data.index

            # Membuat tabel pertama: Original Data dan Training Data
            train_data_table = pd.DataFrame({
                'Tanggal': dates[:len(trainPredict) + look_back],
                'Original Data': scaler_single.inverse_transform(dataset[:len(trainPredict) + look_back, analysis_columns.index(kolom_prediksi)].reshape(-1, 1)).flatten(),
                'Training Data': np.append([np.nan] * look_back, trainPredict.flatten())
            })

            # Membuat tabel kedua: Original Data dan Testing Data
            test_data_table = pd.DataFrame({
                'Tanggal': dates[len(trainPredict) + (look_back * 2): len(trainPredict) + (look_back * 2) + len(testPredict)],
                'Original Data': scaler_single.inverse_transform(dataset[len(trainPredict) + (look_back * 2): len(trainPredict) + (look_back * 2) + len(testPredict), analysis_columns.index(kolom_prediksi)].reshape(-1, 1)).flatten(),
                'Testing Data': testPredict.flatten()
            })

            # Menampilkan tabel dalam kolom yang berbeda
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Tabel: Original Data dan Training Data')
                st.write(train_data_table)

            with col2:
                st.subheader('Tabel: Original Data dan Testing Data')
                st.write(test_data_table)

            # Plot hasil prediksi pada data training
            trainPredictPlot = np.empty_like(dataset)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, analysis_columns.index(kolom_prediksi)] = trainPredict.flatten()

            # Calculate start index for test predictions in testPredictPlot
            start_index = len(trainPredict) + (look_back * 2)

            # Calculate end index for test predictions in testPredictPlot
            end_index = start_index + len(testPredict)

            # Plot data asli
            plt.plot(data.index, scaler_single.inverse_transform(dataset[:, analysis_columns.index(kolom_prediksi)].reshape(-1, 1)), label='Original Data')

            # Plot prediksi pada data training (biru)
            trainPredictPlot = np.empty_like(dataset)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back-1:len(trainPredict)+look_back-1, analysis_columns.index(kolom_prediksi)] = trainPredict.flatten()

            # Atur indeks untuk memplot prediksi pada data training
            train_index = range(look_back-1, look_back-1 + len(trainPredict))

            plt.plot(data.index[:len(train_index)], trainPredictPlot[train_index, analysis_columns.index(kolom_prediksi)], label='Train Prediction', color='green')

            # Plot prediksi pada data testing (merah)
            testPredictPlot = np.empty_like(dataset)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[start_index:end_index, analysis_columns.index(kolom_prediksi)] = testPredict.flatten()

            plt.plot(data.index[start_index:end_index], testPredictPlot[start_index:end_index, analysis_columns.index(kolom_prediksi)], label='Test Prediction', color='red')

            # Tambahkan label dan legenda
            plt.xlabel('Date')
            plt.ylabel(kolom_prediksi + ' Price ($)')
            plt.title('Performance LSTM Model')
            plt.legend()

            # Mendapatkan objek gambar (figure)
            fig = plt.gcf()

            # Tampilkan grafik menggunakan st.pyplot() dengan menyertakan objek gambar
            st.pyplot(fig)

            #-------------------------------------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("RMSE")
                    
                # Menghitung MSE untuk data training
                train_mse = mean_squared_error(trainY_inverse[0], trainPredict[:, 0])
                st.write('MSE untuk data training :', train_mse)

                # Menghitung MSE untuk data testing
                test_mse = mean_squared_error(testY_inverse[0], testPredict[:, 0])
                st.write('MSE untuk data testing:', test_mse)

                # Menghitung RMSE untuk data training
                train_rmse = np.sqrt(train_mse)
                st.write('RMSE untuk data training:', train_rmse)

                # Menghitung RMSE untuk data testing
                test_rmse = np.sqrt(test_mse)
                st.write('RMSE untuk data testing:', test_rmse)

                # Cetak nilai maksimum dan minimum dari harga saham
                max_price = data[kolom_prediksi].max()
                min_price = data[kolom_prediksi].min()

                # Cetak rentang data
                data_range = max_price - min_price
                st.write("Rentang data harga saham dalam ",years," tahun terakhir:", data_range)

                # RMSE dan rentang data
                RMSE_TEST = test_rmse
                RMSE_TRAIN = train_rmse
                rentang_data = data_range

                # Menghitung persentase RMSE
                persentase_RMSE_TRAIN = (RMSE_TRAIN / rentang_data) * 100
                persentase_RMSE_TEST = (RMSE_TEST / rentang_data) * 100

                # Menampilkan hasil
                st.write("Persentase RMSE Training dari rentang data:", persentase_RMSE_TRAIN, "%")
                st.write("Persentase RMSE Testing dari rentang data:", persentase_RMSE_TEST, "%")

                def calculate_mape(actual, predicted):
                    return np.mean(np.abs((actual - predicted) / actual)) * 100

            with col2:
                st.header("MAPE")
                
                # Menghitung MAPE untuk training data
                train_mape = calculate_mape(trainY_inverse[0], trainPredict[:, 0])
                st.write('MAPE untuk Training Data:', train_mape ,'%')

                # Menghitung MAPE untuk testing data
                test_mape = calculate_mape(testY_inverse[0], testPredict[:, 0])
                st.write('MAPE untuk Testing Data:', test_mape ,'%')

            with col3:
                st.header("Pengujian R-squared")
                
                def calculate_r_squared(actual, predicted):
                    residual_sum_of_squares = np.sum((actual - predicted) ** 2)
                    total_sum_of_squares = np.sum((actual - np.mean(actual)) ** 2)
                    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
                    return r_squared

                # Menghitung R-squared untuk data training
                train_r_squared = calculate_r_squared(trainY_inverse[0], trainPredict[:, 0])
                st.write('R-squared untuk Training Data:', train_r_squared)

                # Menghitung R-squared untuk data testing
                test_r_squared = calculate_r_squared(testY_inverse[0], testPredict[:, 0])
                st.write('R-squared untuk Testing Data:', test_r_squared)
            # Simpan hasil ke tabel result
            save_result(st.session_state.username, stock_symbol , rentan_waktu, len(data), kolom_prediksi, persentase_train_size, epochs, batch_size, test_rmse, test_mape, test_r_squared)
            
            # Plot data prediksi masa depan

            last_sequence = dataset[-look_back:]
            future_predictions = []

            for _ in range(selected_days):
                next_prediction = model.predict(last_sequence.reshape(1, look_back, len(analysis_columns)))
                future_predictions.append(next_prediction[0, 0])
                last_sequence = np.append(last_sequence[:, 1:], next_prediction, axis=1)

            future_predictions = scaler_single.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            future_dates = pd.date_range(start=data.index[-1], periods=selected_days + 1, closed='right')

            future_data = pd.DataFrame({
                'Date': future_dates,
                'Future Predictions': future_predictions.flatten()
            })

            st.write(future_data)

if __name__ == '__main__':
    main()

#by raihan nor f. [politeknik negeri tanah laut]