from random import randrange, randint


class Matrix(object):
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for x in range(self.cols)] for y in range(self.rows)]

    # Randomize the newly created matrices
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = randrange(0, 1)

    # Print the matrix
    def show(self):
        print(self.data)

    # From array to matrix object
    @staticmethod
    def from_array(array):
        m = Matrix(len(array), 1)
        for i in range(len(array)):
            m.data[i][0] = array[i]
        return m

    # From matrix object to array
    def to_array(self):
        array = []
        for i in range(self.rows):
            for j in range(self.cols):
                array.append(self.data[i][j])
        return array

    # Add a given n number or a given matrix to matrix
    def add(self, n):
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    # Subtract a given n number or a given matrix to matrix
    @staticmethod
    def subtract(a, b):
        out = Matrix(a.rows, a.cols)
        for i in range(out.rows):
            for j in range(out.cols):
                out.data[i][j] = a.data[i][j] - b.data[i][j]
        return out

    # Multiply a given n number (scalar) or a given matrix to matrix
    def multiply(self, n):
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    # Dot product of 2 matrices (first matrix cols must match second matrix rows)
    @staticmethod
    def dot(m1, m2):
        if m1.cols != m2.rows:
            print("Matrices must match!")

            return None
        else:
            out = Matrix(m1.rows, m2.cols)
            for i in range(out.rows):
                for j in range(out.cols):
                    sum = 0
                    for k in range(m1.cols):
                        sum += m1.data[i][k] * m2.data[k][j]
                    out.data[i][j] = sum

            return out

    # Transpose given matrix
    @staticmethod
    def transpose(matrix):
        out = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                out.data[j][i] = matrix.data[i][j]
        return out

    # Map given matrix with given function (sigmoid, derivative...)
    @staticmethod
    def map(matrix, func):
        out = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                out.data[i][j] = func(matrix.data[i][j])
        return out
