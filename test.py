loginid = "loginid"
passwd = "passwd"

# List of car makers
car_make = ['BMW', 'Volkswagen', 'Volvo', 'Opel', 'Hyundai','Skoda', 'KIA', 'Audi', 'Ford', 'Mazda', 'Peugeot']
# Dictionary of different models
car_model = {'Volkswagen': ['Passat'], 'Volvo': ['V60', 'S60', 'V70', 'S70', 'V90', 'S90'], 'Opel': ['Insignia'], 'Hyundai': ['i40'], 'Skoda': ['Superb'], 'KIA': ['Optima'], 'Audi': ['A4'], 'Toyota': ['Avensis'], 'Ford': ['Mondeo'], 'Mazda': ['6'], 'Peugeot': ['508'], 'BMW': ['3 - Sarja (Kaikki)']}
# List of model years
car_year = ['2009', '2010', '2011', '2012', '2013','2014', '2015', '2016', '2017', '2018', '2019', '2020']

#Scrape data from Nettiauto.com and write to a csv
X = []
for make in car_make:
    for model in car_model[make]:
        for year in car_year:
            if os.path.exists('%s_%s_%s.csv' % (make, model, year)):
                X_tmp = pd.read_csv('%s_%s_%s.csv' % (make, model, year))
                X_tmp = X_tmp.loc[:, ~X_tmp.columns.str.contains('^Unnamed')]
            else:
                try:
                    driver, X_tmp = scrape_car_data(driver, car_make = make, car_model = model, car_year = year)
                except ValueError:
                    pass
                except NameError:
                    driver = login_webpage(loginid, passwd)
                    driver, X_tmp = scrape_car_data(driver, car_make = make, car_model = model, car_year = year)
                X_tmp.to_csv('%s_%s_%s.csv' % (make, model, year))
            X.append(X_tmp)

driver.quit()
del driver
X = pd.concat(X, ignore_index=True, sort=False)

X_scaled, y_scaled, xscale, xmin, yscale, ymin = preprocess_data(X, extra_variables=[['Hybrid', 'plug', 'phev','gte'],['webasto'],['(3.0)'],['(2.5)'],['(2.2)'],['(2.0)'],['(1.8)'],['(1.6)'],['(1.5)'],['(1.4)'],['(1.2)'],['(1.0)']], scalers=True)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1)

model = models.Sequential()
model.add(layers.Dense(80, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test))
