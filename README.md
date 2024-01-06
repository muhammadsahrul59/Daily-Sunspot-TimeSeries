# Time Series Forecasting - Daily Sunspot (1850 - 2023)✨

## Project Description
Embark on a journey through time with the Time Series Forecasting project, focusing on predicting daily sunspot activity from 1850 to 2023. This project harnesses the power of Long Short-Term Memory (LSTM) networks to capture temporal dependencies and provide accurate predictions.

You can view or download the dataset in Kaggle : [Daily Sunspots Dataset (1850 - 2023)]( https://www.kaggle.com/datasets/patrickfleith/daily-sunspots-dataset )

## Key Features
- **Large Dataset**: Boasting over 10,000 samples of daily sunspot data, this project provides an extensive dataset for robust time series forecasting.
- **LSTM Model Architecture**: The model incorporates LSTM layers to effectively capture long-term dependencies within the time series data, enabling accurate predictions.
- **Validation Set Allocation**: Strategically divides the dataset, allocating 20% for a validation set to ensure proper model evaluation during training.
- **Sequential Model Design**: Adhering to the sequential model design, the architecture ensures a smooth flow of information, enhancing the model's ability to learn from historical patterns.
- **Learning Rate Optimization**: Implements learning rate tuning within the optimizer to enhance model convergence and speed up the training process.
- **MAE < 10% of Data Scale**: Aiming for precision, the Mean Absolute Error (MAE) of the model is set to be less than 10% of the data scale, ensuring accurate predictions on the daily sunspot time series.

## Run Google Colab
```
- Download the file above
- open https://colab.research.google.com/
- Upload the file Daily_Sunspot_TimeSeries.ipynb
```
or you can open it with this Google Colab Link below :

[![](https://img.shields.io/badge/Google%20colab-Open-FFC000.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAB8lBMVEUAAAD/7QL/6w7/6gz/6w3/6Qj/6Qj/6wj/6gf/6w//6w7/6xP/7gP/2hr/2xX/2jD/z0j/uHX/1zn/3Sz/3Sv/1zn/r4L/zE3/2jL/4ST/4Cf/0RP//zH/3S//2jL/2jL/2zH/3iv/3in/2zD/2jL/2jL/3C///0b/1B3/yxL/xAj/0SH/2DT/2TX/2TX/3C7/2jP/2DX/2DX/2DX/1zD/zRb/YwD/wgf/wQf/xAr/1iX/3C//3C//2zH/2Db/2jP/3C7/2zD/zBf/wgj/wgf/wgf/wQf/wQf/0ST/2y7/5B3/2TT/2TT/3iv/2y//zSD/wQb/wQf/wgf/wgf/wQf/wQf/zx7/3Cz/4x//2TT/2TX/4TD/zRb/wQf/wgf/wQf/wgf/wgf/wQf/wwr/1iT/3S//3C7/2zH/1zH/zBb/wQT/wgf/wQf/wQf/wgf//4r/xAj/0CD/2DT/2TX/2TX/3C7/zBb/wgj/wQf/wQf/wQf/wgf/xgf/0BL/6Rz/3C//2jL/2jP/2zH/3iz/vwD/wgf/wgf/wgf/wgf/xAf/wgf/2Rr/2hf/2TH/t3f//wD/1D//3S7/wgX/wAf/0wj/uQf/wQf/wwf/wgf/5gX/5h//6BT/5xb/6Q//6RD/xAf/xAf/xQf/xAf/xQf/xAf///9tAmifAAAApXRSTlMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATiFmGgWFGaZi0EDAABa6v/5+32y//r/8WoBJ9viYD9wnv+kPlvc5TRg/XsAAArH4yEAAGz+cmL9dgAACsjhHgAAZ/50LOHcUzRbm/+VLUjT6TcAZvH89Pd5uv/x+fRxAgADRJSldRsYcKOVSAQAAAAAAAEAAAAAAQAAAAAAAAAAAAAAAAAAAABydb0CAAAAAWJLR0SlLrlKLwAAAAd0SU1FB+QIEQUaC7jgabcAAAABb3JOVAHPoneaAAAAwUlEQVQI12NgQAOMTMwsrGzsHJxc3DwMDLx8/AKCQsIiomLiEpJSDNIysnLyCopKyiqqauoaDJpa2jq6evoGhkbGJqZmDOYWllbWNrZ29g6OTs4uDK5u7h6eXt4+vn7+AYFBDMEhoWHhEZFR0TGxcfEJDIlJySmpaekZmVnZObl5DPkFhUXFJaVl5RWVVdU1DLV19Q2NTc0trW3tHZ1dDN09vX39EyZOmjxl6rTpMxgYZs6aPWfuvPkLFi5avATdAwCl1jTbaxUL4wAAAF5lWElmSUkqAAgAAAADABIBAwABAAAAAQAAADEBAgANAAAAMgAAAGmHBAABAAAAQAAAAAAAAABQaG90b3MgMi43LjAAAAIAAqAJAAEAAADTAAAAA6AJAAEAAADTAAAAAAAAAJOs9JgAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjAtMDgtMTdUMDU6MjY6MTErMDA6MDBd4jn+AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIwLTA4LTE3VDA1OjI2OjExKzAwOjAwLL+BQgAAABh0RVh0ZXhpZjpFeGlmSW1hZ2VMZW5ndGgAMjExdjfLjwAAABd0RVh0ZXhpZjpFeGlmSW1hZ2VXaWR0aAAyMTHrqE6dAAAAEnRFWHRleGlmOkV4aWZPZmZzZXQANjTZeQZNAAAAGnRFWHRleGlmOlNvZnR3YXJlAFBob3RvcyAyLjcuMIcagsYAAAAASUVORK5CYII=)](https://drive.google.com/file/d/1W5WPd22ch5CuZHEIzFSwdOkRG8bPMeMe/view?usp=sharing)

Feel free to delve into the nuances of time series forecasting and contribute to the project's improvement. Your engagement and insights are highly appreciated! :pray:
