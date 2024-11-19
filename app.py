import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Заголовок
st.title("Прогноз споживання енергії")

# Завантаження файлу
uploaded_file = st.file_uploader("Завантажте файл CSV з даними", type="csv")
if uploaded_file:
    # Завантаження та обробка даних
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data['date'], dayfirst=True)
    data.set_index('date', inplace=True)
    
    # Вибір числових стовпців
    data_numeric = data.select_dtypes(include=['float64', 'int64'])
    
    # Ресемплінг даних за годинами
    data_hourly = data_numeric.resample('H').mean()

    # Відображення перших рядків даних
    st.write("Перегляд даних:")
    st.dataframe(data.head())

    # Декомпозиція часового ряду
    st.subheader("Декомпозиція часового ряду")
    if 'Usage_kWh' in data_hourly.columns:
        decomposition = seasonal_decompose(data_hourly['Usage_kWh'], model='additive', period=24)
        fig = decomposition.plot()
        st.pyplot(fig)
    else:
        st.warning("Стовпець 'Usage_kWh' відсутній у даних.")

    # Кореляційна матриця
    st.subheader("Кореляційна матриця")
    if not data_numeric.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Немає числових стовпців для аналізу.")

    # Побудова та тренування моделі LSTM
    st.subheader("Прогнозування з використанням LSTM")
    if 'Usage_kWh' in data_hourly.columns:
        # Підготовка даних для моделі
        series = data_hourly['Usage_kWh'].dropna().values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_series = scaler.fit_transform(series.reshape(-1, 1))
        
        train_size = int(len(scaled_series) * 0.8)
        train, test = scaled_series[:train_size], scaled_series[train_size:]

        def create_sequences(data, time_steps=1):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:i + time_steps])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)

        time_steps = 24
        X_train, y_train = create_sequences(train, time_steps)
        X_test, y_test = create_sequences(test, time_steps)

        # Створення моделі
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
            LSTM(50, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Навчання моделі
        with st.spinner("Триває навчання моделі..."):
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # Прогнозування
        y_pred = model.predict(X_test)
        y_test_inverse = scaler.inverse_transform(y_test)
        y_pred_inverse = scaler.inverse_transform(y_pred)

        # Оцінка точності
        mse = mean_squared_error(y_test_inverse, y_pred_inverse)
        st.write(f"Середньоквадратична помилка (MSE): {mse:.2f}")

        # Візуалізація результатів
        st.subheader("Прогнозовані vs Реальні дані")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(y_test_inverse)), y_test_inverse, label="Реальні дані")
        ax.plot(range(len(y_pred_inverse)), y_pred_inverse, label="Прогноз", linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Прогноз на майбутнє
        st.subheader("Прогноз на майбутнє")
        n_days = st.slider("Виберіть кількість днів для прогнозу:", 1, 14, 7)
        n_hours = n_days * 24
        input_sequence = scaled_series[-time_steps:].reshape(1, time_steps, 1)
        predictions = []

        for _ in range(n_hours):
            predicted_value = model.predict(input_sequence, verbose=0)
            predictions.append(predicted_value[0, 0])
            input_sequence = np.append(input_sequence[:, 1:, :], predicted_value.reshape(1, 1, 1), axis=1)

        predicted_values_inverse = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        forecast_index = pd.date_range(data_hourly.index[-1], periods=n_hours + 1, freq='H')[1:]

        # Візуалізація прогнозу
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data_hourly.index, data_hourly['Usage_kWh'], label="Реальні дані")
        ax.plot(forecast_index, predicted_values_inverse, label=f"Прогноз на {n_days} днів", linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Для прогнозування потрібен стовпець 'Usage_kWh'.")



