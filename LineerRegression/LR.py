# Lineer regresyon modeli sınıfı
class LinearRegression:
    def __init__(self, learning_rate=0.000005, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m1 = 1
        self.m2 = 2
        self.b = 0

    # Modeli eğitmek için uygun fonksiyon
    def fit(self, X, Y, Z):
        n = len(X)
        for _ in range(self.epochs):
            d_m1 = -(2/n) * sum(X[i] * (Z[i] - self.predict_point(X[i], Y[i])) for i in range(n))
            d_m2 = -(2/n) * sum(Y[i] * (Z[i] - self.predict_point(X[i], Y[i])) for i in range(n))
            d_b = -(2/n) * sum(Z[i] - self.predict_point(X[i], Y[i]) for i in range(n))

            # Parametreleri güncelle
            self.m1 -= self.learning_rate * d_m1
            self.m2 -= self.learning_rate * d_m2
            self.b -= self.learning_rate * d_b

    # Tek bir veri noktası için tahminde bulun
    def predict_point(self, x, y):
        return self.m1 * x + self.m2 * y + self.b

    # Verilen girdi matrisi için tahminler yap
    def predict(self, X, Y):
        return [self.predict_point(X[i], Y[i]) for i in range(len(X))]
