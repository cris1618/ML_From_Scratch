
# Table of Contents

1.  [Introduction](#orgfc1a618)
2.  [Code](#orge8c81f2)



<a id="orgfc1a618"></a>

# Introduction

In this notebook, I will implement a Linear Regression algorithm 
from scratch using Python, without relying on any external libraries.


<a id="orge8c81f2"></a>

# Code

The goal is to minimize the error function,
the error function is defined as:

\[
E = \frac{1}{n} \sum (y_i - (mx + b))^2
\]

We want to find the values that minimize `E`. To do that, we can 
tweak the parameters `m` and `b` to minimize `E`. Therefore, we need 
to find the partial derivatives with respect to `m` and `b`.

This will allow us to change the two parameters using these updating formulas:

\[
m = m - L \cdot \frac{\partial E}{\partial m}
\]
\[
b = b - L \cdot \frac{\partial E}{\partial b}
\]

Now let's see the code!

    import matplotlib.pyplot as plt
    import pandas as pd
    import random
    
    num_rows = 100 
    # Creating random data 
    grades = [random.randint(1,100) for _ in range(num_rows)]
    time = [random.randint(1,100) for _ in range(num_rows)]
    
    data = {"grades":grades, "studytime": time}
    df = pd.DataFrame(data)
    
    # Define the Loss function (Not used, only for learning purposes)
    def loss_function(m, b, points):
        total_error = 0
        for i in range(len(points)):
    	x = points.iloc[i].studytime
    	y = points.iloc[i].grades
    	total_error += (y - (m*x +b))**2 
        return total_error / float(len(points))
    
    # Define the Gradient Descent
    def gradient_descent(m_now, b_now, points, L):
        m_grad = 0
        b_grad = 0
    
        n = len(points)
        # Using the partial derivatives that we already found
        for i in range(n):
    	x = points.iloc[i].studytime
    	y = points.iloc[i].grades
    	m_grad += -(2/n)*(x)*(y - (m_now*x+b_now))
    	b_grad += -(2/n)*(y - (m_now*x+b_now))
    
        m = m_now - m_grad*L
        b = b_now - b_grad*L
        return m, b
    
    # Apply the Linear Regresion
    m = 0
    b = 0
    L = 0.0001
    epochs = 1000
    
    for i in range(epochs):
        if i % 100 == 0:
           print(f"Epoch: {i}")
        m, b = gradient_descent(m, b, df, L)
    
    print(m,b)

    Epoch: 0
    Epoch: 100
    Epoch: 200
    Epoch: 300
    Epoch: 400
    Epoch: 500
    Epoch: 600
    Epoch: 700
    Epoch: 800
    Epoch: 900
    0.7233503080487481 2.505659317798391

Visualize the result with the parameters `m` and `b` learned by the model.

    import matplotlib.pyplot as plt
    import pandas as pd
    import random
    
    num_rows = 100 
    # Creating random data 
    grades = [random.randint(1,100) for _ in range(num_rows)]
    time = [random.randint(1,100) for _ in range(num_rows)]
    
    data = {"grades":grades, "studytime": time}
    df = pd.DataFrame(data)
    
    # Define the Loss function (Not used, only for learning purposes)
    def loss_function(m, b, points):
        total_error = 0
        for i in range(len(points)):
    	x = points.iloc[i].studytime
    	y = points.iloc[i].grades
    	total_error += (y - (m*x +b))**2 
        return total_error / float(len(points))
    
    # Define the Gradient Descent
    def gradient_descent(m_now, b_now, points, L):
        m_grad = 0
        b_grad = 0
    
        n = len(points)
        # Using the partial derivatives that we already found
        for i in range(n):
    	x = points.iloc[i].studytime
    	y = points.iloc[i].grades
    	m_grad += -(2/n)*(x)*(y - (m_now*x+b_now))
    	b_grad += -(2/n)*(y - (m_now*x+b_now))
    
        m = m_now - m_grad*L
        b = b_now - b_grad*L
        return m, b
    
    # Apply the Linear Regresion
    m = 0
    b = 0
    L = 0.0001
    epochs = 1000
    
    for i in range(epochs):
        if i % 100 == 0:
           print(f"Epoch: {i}")
        m, b = gradient_descent(m, b, df, L)
    
    print(m,b) 
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax.scatter(df["grades"], df["studytime"], color = "black")
    ax.plot(df["studytime"], [m * x + b for x in df["studytime"]], color="red")
    ax.set_xlabel("Grades")
    ax.set_ylabel("Time Spent")
    ax.set_title("Correlation between Grades and Time Spent")
    ax.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("lgs.png")
    plt.show()

![img](lgs.png)

