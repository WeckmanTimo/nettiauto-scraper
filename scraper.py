import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from bs4 import BeautifulSoup as bs
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

url = "https://www.nettiauto.com/statVehicle.php"

def login_webpage(login_id, login_passwd):
    # Accept cookies and login
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(executable_path=r'E://chromedriver_win32/chromedriver.exe', options=options)
    driver.get(url)
    driver.implicitly_wait(1)
    # Click cookies
    driver.switch_to.frame(driver.find_element_by_id("gdpr-consent-notice"))
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(@id, "save")]')))
    driver.find_element_by_xpath('//button[contains(@id, "save")]').click()
    driver.switch_to.default_content()
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.ID, 'fixed_footer')))
    # Fill in the login form
    login = driver.find_elements_by_name("loginid")[-1]
    passwd = driver.find_elements_by_name("passwd")[-1]
    login.send_keys(login_id)
    passwd.send_keys(login_passwd)
    driver.find_element_by_id("loginSubmit").click()
    return driver

def scrape_car_data(driver, car_make, car_model, car_year):
    """
    Scrapes the Nettiauto.com website for data on past sales
    Args: 
        driver: webdriver object, opened at the correct url
        car_make: string, the car manufacturer, e.g. Volvo, Audi, Volkswagen
        car_model: string, model of a car, should correspond to the manufacturer for the parsing to work
        car_year: string, year of the model
    """
    # Make a search based on the args
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'sid_make')))
    make = driver.find_element_by_name("sid_make")
    model = driver.find_element_by_name("sid_model")
    year = driver.find_element_by_name("syear")
    search = driver.find_element_by_name("search")
    Select(make).select_by_visible_text(car_make)
    Select(model).select_by_visible_text(car_model)
    Select(year).select_by_visible_text(car_year)
    search.click()
    
    # Scrape the search page of data
    soup = bs(driver.page_source)
    datatable = soup.find('div', {"id": "vehicle_statistics_heading"})
    car_data = pd.DataFrame([item.text.strip() for item in datatable.findAll(attrs={'class' : 'col1'})])
    car_data.columns = car_data.iloc[0]
    car_data = car_data.drop(car_data.index[0])
    year_data = pd.DataFrame([item.text.strip() for item in datatable.findAll(attrs={'class' : 'col2'})])
    year_data.columns = year_data.iloc[0]
    year_data = year_data.drop(year_data.index[0])
    mileage_data = pd.DataFrame([item.text.split('€')[0].strip().replace(' ','') for item in datatable.findAll(attrs={'class' : 'col3'})])
    mileage_data = pd.DataFrame(mileage_data.values.reshape(-1,2))
    mileage_data.columns = mileage_data.iloc[0]
    mileage_data = mileage_data.drop(mileage_data.index[0])
    date_data = pd.DataFrame([item.text.strip() for item in datatable.findAll(attrs={'class' : 'col4'})])
    date_data.columns = date_data.iloc[0]
    date_data = date_data.drop(date_data.index[0])

    # Iterate all the pages
    try:
        last_page = driver.find_elements_by_name("1")[-1].text
    except IndexError:
        last_page = 1
    for i in range(2, int(last_page)+1):
        WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, '%d' % i)))
        driver.find_element_by_name('%d' % i).click()

        soup = bs(driver.page_source)
        datatable = soup.find('div', {"id": "vehicle_statistics_heading"})
        car_data_tmp = pd.DataFrame([item.text.strip() for item in datatable.findAll(attrs={'class' : 'col1'})])
        car_data_tmp.columns = car_data_tmp.iloc[0]
        car_data_tmp = car_data_tmp.drop(car_data_tmp.index[0])
        year_data_tmp = pd.DataFrame([item.text.strip() for item in datatable.findAll(attrs={'class' : 'col2'})])
        year_data_tmp.columns = year_data_tmp.iloc[0]
        year_data_tmp = year_data_tmp.drop(year_data_tmp.index[0])
        mileage_data_tmp = pd.DataFrame([item.text.split('€')[0].strip().replace(' ','') for item in datatable.findAll(attrs={'class' : 'col3'})])
        mileage_data_tmp = pd.DataFrame(mileage_data_tmp.values.reshape(-1,2))
        mileage_data_tmp.columns = mileage_data_tmp.iloc[0]
        mileage_data_tmp = mileage_data_tmp.drop(mileage_data_tmp.index[0])
        date_data_tmp = pd.DataFrame([item.text.strip() for item in datatable.findAll(attrs={'class' : 'col4'})])
        date_data_tmp.columns = date_data_tmp.iloc[0]
        date_data_tmp = date_data_tmp.drop(date_data_tmp.index[0])

        car_data = pd.concat([car_data, car_data_tmp], ignore_index=True, sort=False)
        year_data = pd.concat([year_data, year_data_tmp], ignore_index=True, sort=False)
        mileage_data = pd.concat([mileage_data, mileage_data_tmp], ignore_index=True, sort=False)
        date_data = pd.concat([date_data, date_data_tmp], ignore_index=True, sort=False)

    makemodel_data = pd.DataFrame([[car_make, car_model]])
    makemodel_data = pd.concat([makemodel_data] * car_data.shape[0], ignore_index=True, sort=False)
    makemodel_data.columns = ['Merkki', 'Malli']
    # Convert dates into a float
    dates = date_data['Myyntipvm'].str.split('-',expand=True).astype('float64')
    date_data['Myyntipvm'] = dates.iloc[:,0] / 12. + dates.iloc[:,1]
    # Merge all the data into a one dataframe
    X = pd.merge(makemodel_data, mileage_data, left_index=True, right_index=True)
    X = pd.merge(X, year_data, left_index=True, right_index=True)
    X = pd.merge(X, date_data, left_index=True, right_index=True)
    X = pd.merge(X, car_data, left_index=True, right_index=True)
    return driver, X

def preprocess_data(X, extra_variables=[], scalers=False):
    """
    Preprocesses the dataframe by scaling the data (MinMax-scaling) and making dummy variables for category-variables
    Args:
        X, pandas DataFrame, the scraped data from Nettiauto.com
        extra_variables, list, keywords that are to be parsed out from the car info text, 
            e.g. "plug" or "hybrid" for any hybrid etc.
            contains strings or lists of strings, i.e. [['plug', 'hybrid'], 'webasto']
        scalers, True/False, whether or not the MinMax-slacer scaling parameters are returned
    Returns:
        X, pandas DataFrame, scaled dataset
        y, pandas Series, target values
        if scalers=True:
            xscale, dictionary, scaling parameters for continuous X-variables
            xmin, dictionary, scaling parameters for continuous X-variables
            yscale, float, scaling parameters for continuous y-variables
            ymin, float, scaling parameters for continuous y-variables
    """
    cont_variables = ['Hinta', 'Mittarilukema', 'Vuosi', 'Myyntipvm']
    categ_variables = ['Merkki', 'Malli', 'Merkki & Malli']


    extra_var = np.zeros((X.shape[0], len(extra_variables)))
    for i, icar in enumerate(X['Merkki & Malli']):
        for j, test in enumerate(extra_variables):
            if type(test) == list:
                for test_case in test:
                    if test_case.lower() in icar.lower():
                        extra_var[i,j] = 1
                        break
            else:
                if test.lower() in icar.lower():
                    extra_var[i,j] = 1
    extra_var = pd.DataFrame(extra_var)
    extra_var.columns = [i  if (type(i) != list) else i[0] for i in extra_variables]
    categ_variables.remove('Merkki & Malli')
    del X['Merkki & Malli']
    cont_variables.remove('Myyntipvm')
    del X['Myyntipvm']

    # Scale continuous variables
    xscale = {}
    xmin = {}
    for var in cont_variables:
        X[var] = pd.to_numeric(X[var], errors='coerce')
        X[var] = X[var].replace(np.nan, 0)
        try:
            xscale[var] = 1 / (np.max(X[var])-np.min(X[var]))
        except ZeroDivisionError:
            xscale[var] = 1.
        xmin[var] = np.min(X[var])
        X[var] = (X[var] - xmin[var]) * xscale[var]
    
    # If only one make/model is used, remove as a variable
    for var in categ_variables:
        if X[var].nunique() == 1:
            del X[var]
        else:
            X = pd.concat([X, pd.get_dummies(X[var], prefix=var)], axis=1)
            del X[var]
        X = pd.merge(X, extra_var, left_index=True, right_index=True)
    
    # Target
    y = X['Hinta']
    yscale = xscale['Hinta']
    ymin = xmin['Hinta']
    cont_variables.remove('Hinta')
    del X['Hinta']
    
    if scalers:
        return X, y, xscale, xmin, yscale, ymin
    else:
        return X, y
