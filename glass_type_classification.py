import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glasstype = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glasstype = glasstype[0]
    if glasstype == 1:
        return "Building windows float processed"
    elif glasstype == 2:
        return "Building windows non float processed"
    elif glasstype == 3:
        return "vehicle windows float processed".upper()
    elif glasstype == 4:
        return "vehicle windows non float processed".upper()
    elif glasstype == 5:
        return "containers".upper()
    elif glasstype == 6:
        return "tableware".upper()
    else:
        return 'Headlamp'
st.title('Glass Type Prediction Web App')
st.subheader('-Sidartha Ramprakash')
st.sidebar.title('Glass Type Prediction Web App')
if st.sidebar.checkbox('Show raw data'):
    st.subheader('Glass Type Data set')
    st.dataframe(glass_df)
st.sidebar.subheader('Visualization Selecter')
plot_list = st.sidebar.multiselect('Select the charts/plots', ('Line Chart', 'Area Chart', 'Count Plot', 'Correlation Heatmap', 'Piechart', 'Boxplot'))
if 'Line Chart' in plot_list:
    st.subheader('Linechart')
    st.line_chart(glass_df)
if 'Area Chart' in plot_list:
    st.subheader('Area Chart')
    st.area_chart(glass_df) 
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Correlation Heatmap' in plot_list:
    st.subheader('Correlation Heatmap')
    plt.figure(figsize = (10, 5)) 
    sns.heatmap(glass_df.corr(), annot = True)
    st.pyplot()
if 'Count Plot' in plot_list:
    st.subheader('Count Plot')
    plt.figure(figsize = (10, 5)) 
    sns.countplot(x = 'GlassType', data = glass_df)
    st.pyplot()
if 'Piechart' in plot_list:
    st.subheader('Piechart')
    plt.figure(figsize = (10, 5))
    pie_val = glass_df['GlassType'].value_counts()
    plt.pie(pie_val, labels = pie_val.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(0.06, 0.12, 6))
    st.pyplot()
if 'Boxplot' in plot_list:
    st.subheader('Boxplot')
    column = st.sidebar.selectbox('Select The column for Box Plot', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
    plt.figure(figsize = (10, 5)) 
    sns.boxplot(glass_df[column])
    st.pyplot()
st.sidebar.subheader('Select your values')
ri = st.sidebar.slider('Input RI', float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider('Input Na', float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider('Input Mg', float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider('Input Al', float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider('Input Si', float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider('Input K', float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider('Input Ca', float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
fe = st.sidebar.slider('Input Fe', float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))
ba = st.sidebar.slider('Input Ba', float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))
if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()
if classifier =='Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf= RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rf_clf, X_test, y_test)
        st.pyplot()
if classifier =='Logistic Regression':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    max_iter_input = st.sidebar.number_input('Maximum iterations', 10, 10000, step = 10)
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        log_reg= LogisticRegression(C = c_value, max_iter = max_iter_input)
        log_reg.fit(X_train,y_train)
        accuracy = log_reg.score(X_test, y_test)
        glass_type = prediction(log_reg, ri, na, mg, al, si, k, ca, ba, fe)
        st.write('The type of glass predicted is', glass_type)
        st.write('Accuracy of model is', accuracy.round(2))
        plot_confusion_matrix(log_reg, X_test, y_test)
        st.pyplot()