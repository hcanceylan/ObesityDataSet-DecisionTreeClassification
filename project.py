import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import mmap
from io import StringIO
from threading import Thread
import numpy as np
from sklearn import metrics

def arrange_data(mydata): #arranges incoming big datas to numeric values 
    
    mydata["Gender"] = [1 if index == "Female" else 2 for index in mydata["Gender"]]
    
    mydata["Age"] = [1 if index<=20 else 2 if index>20 and index<=40 else 3 if index>40 and index<=60 else 4 for index in mydata["Age"]]
    
    mydata["Height"] = [1 if index<1.5 else 2 if index>=1.5 and index<1.7 else 3 if index>=1.7 and index<1.9 else 4 for index in mydata["Height"]]
    
    mydata["Weight"] = [1 if index<40 else 2 if index>=40 and index<60 else 3 if index>=60 and index<85 else 4 for index in mydata["Weight"]]

    mydata["family_history_with_overweight"] = [1 if index == "yes" else 2 for index in mydata["family_history_with_overweight"]]

    mydata["FAVC"] = [1 if index == "yes" else 2 for index in mydata["FAVC"]]
    
    mydata["FCVC"] = [1 if index>=0.5 and index<1.5 else 2 if index>=1.5 and index<2.5 else 3 if index>=2.5 and index<3.5 else 4 for index in mydata["FCVC"]]
    
    mydata["NCP"] = [1 if index>=0.5 and index<1.5 else 2 if index>=1.5 and index<2.5 else 3 if index>=2.5 and index<3.5 else 4 for index in mydata["NCP"]]
    
    mydata["CAEC"] = [1 if index == "Always" else 2 if index == "Frequently" else 3 if index == "Sometimes" else 4 for index in mydata["CAEC"]]

    mydata["SMOKE"] = [1 if index == "Yes" else 2 for index in mydata["SMOKE"]]

    mydata["SCC"] = [1 if index == "Yes" else 2 for index in mydata["SCC"]]
    
    mydata["CH2O"] = [1 if  index<1 else 2 if index>=1 and index<2 else 3 for index in mydata["CH2O"]]
    
    mydata["FAF"] = [1 if index < 0.5 else 2 if index>=0.5 and index<1.5 else 3 if index>=1.5 and index<2.5 else 4 for index in mydata["FAF"]]
    
    mydata["TUE"] = [1 if index < 0.5 else 2 if index>=0.5 and index<1.5 else 3 if index>=1.5 and index<2.5 else 4 for index in mydata["TUE"]]
    
    mydata["CALC"] = [1 if index == "Always" else 2 if index == "Frequently" else 3 if index == "Sometimes" else 4 for index in mydata["CALC"]]

    mydata["MTRANS"] = [1 if index == "Automobile" else 2 if index == "Motorbike" else  3 if index == "Public_Transportation" else 4  if index == "Bike" else 5 for index in mydata["MTRANS"]]

    mydata["NObeyesdad"] = [1 if index == "Normal_Weight" else 2 if index == "Overweight_Level_I" else 3  if index == "Overweight_Level_II" else 4  if index == "Obesity_Type_I" else 5 if index == "Obesity_Type_II" else 6  for index in mydata["NObeyesdad"]]
    
    Y = mydata["NObeyesdad"]

    X = mydata.iloc[:,[i for i in range(16)]]

    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1,1)
    
    return X,Y

if __name__ == "__main__":

    with open("ObesityDataSet_raw_and_data_sinthetic.csv") as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m: #Caches big data into RAM for avoid lots of File I/O
            
            #inputElements = np.array([0.65,21,1.62,64,0.65,0.15,2,3,0.2,0.1,2,0.26,0,1,0.01,0.25]).reshape(1,-1) -- 
            
            
            familyInput =  {"Yes":1,"No":2} #family history
            fastFoodInput =  {"Yes":1,"No":2} #is eat fast food
            fcvc = {"Always":1,"Sometimes":2,"Rarely":3} #frequency of eating vegetables
            mealsInput = {"1":1,"2":2,"3":3,"more than 3":4}#frequency of eationg meals per day
            caec = {"Always":1,"Usually":2,"Sometimes":3,"Rarely":4}#snacking
            smokeInput =  {"Yes":1,"No":2}#smoking
            CH2O = {"Less than one liter":1,"Between 1 and 2 liters":2,"More than 2 liters":3}#water drinking
            SCC =  {"Yes":1,"No":2} #controlling taking calories 
            Physica = {"1 to 2 days" : 1,"3 to 4 days":2,"5 to 6 days":3,"No physical activity":4} #physical acivity 
            Technologia = {"0 to 2 hours":1,"3 to 5 hours":2,"More than 5 hours":3}
            Alkool = {"No consume alcohol":4,"Rarely":3,"Weekly":2,"Daily":1}
            MTrans = {"Public Transportation":3,"Motorbike":2,"Bike":4,"Walking":5,"Automobile":1}
            
            
            mydata=pd.read_csv(StringIO(str(m[:],'utf-8'))) #Converts big datas raw bytes to pandas dataframe
            X,Y = arrange_data(mydata)
              
       
            master = tk.Tk()
            gender = tk.DoubleVar()
            
            sexLabel = tk.Label(master, text='Sex').grid(row=0, column=0, sticky=tk.W)
            tk.Radiobutton(master, text='Male', variable=gender, value=0.35).grid(row=0, column=1, sticky=tk.W)
            tk.Radiobutton(master, text='Female', variable=gender, value=0.65).grid(row=0, column=2, sticky=tk.W)
            l = tk.Label(master, text='age')
            l.grid(row=0, column=3, sticky=tk.W)
            age = tk.Spinbox(master, from_=0, to=100)
            age.grid(row=0, column=4, sticky=tk.W)
            weightLabel = tk.Label(master, text='Weight').grid(row=0, column=5, sticky=tk.W)
            weightEntry = tk.Entry(master)
            weightEntry.grid(row=0, column=6, sticky=tk.W)
            kgLabel = tk.Label(master, text='Kg').grid(row=0, column=7, sticky=tk.W)
            heightLabel = tk.Label(master, text='Height').grid(row=0, column=9, sticky=tk.W)
            
            heightEntry = tk.Entry(master)
            heightEntry.grid(row=0, column=10, sticky=tk.W)
            
            metersLabel = tk.Label(master, text='Meters')
            metersLabel.grid(row=0, column=11, sticky=tk.W)
            
            # first row
            text1Label = tk.Label(master, text='Answer the following question').grid(row=3, column=0, columnspan=3, sticky=tk.EW)
            FamilyLabel = tk.Label(master, text='Family medical history').grid(row=4, column=0, columnspan=3, pady=10, sticky=tk.EW)
            family = ttk.Combobox(master, values=("Yes","No"))
            family.grid(row=5, column=0, columnspan=3, pady=10)
            
            
            intakenLabel = tk.Label(master, text='Intake of food between meals?').grid(row=4, column=4, columnspan=3, pady=10,
                                                                                    sticky=tk.EW)
            caecIntaken = ttk.Combobox(master, values=("Always", "Usually", "Sometimes", "Rarely"))
            caecIntaken.grid(row=5, column=4, columnspan=3,
                                                                                                     pady=10)
            
            physicalLabel = tk.Label(master, text='Frequency of physical activity').grid(row=4, column=8, columnspan=3, pady=10,
                                                                                      sticky=tk.EW)
            physical = ttk.Combobox(master, values=("1 to 2 days", "3 to 4 days", "5 to 6 days", "No physical activity"))
            physical.grid(row=5, column=8, columnspan=3, pady=10)
            
            # swcond row
            
            fastFoodLabel = tk.Label(master, text='Do you eat fast food? ').grid(row=6, column=0, columnspan=3, pady=10, sticky=tk.EW)
            fastFood = ttk.Combobox(master, values=("Yes", "No"))
            fastFood.grid(row=7, column=0, columnspan=3, pady=10)
            
            intakenLabel = tk.Label(master, text='Smoke?').grid(row=6, column=4, columnspan=3, pady=10, sticky=tk.EW)
            smoke = ttk.Combobox(master, values=("Yes", "No"))
            smoke.grid(row=7, column=4, columnspan=3, pady=10)
            
            activityLabel = tk.Label(master, text='Frequency of use of technology devices ').grid(row=6, column=8, columnspan=3,
                                                                                                 pady=10, sticky=tk.EW)
            activity = ttk.Combobox(master, values=("0 to 2 hours", "3 to 5 hours", "More than 5 hours"))
            activity.grid(row=7, column=8,columnspan=3,pady=10)
            
            # third row
            
            consumptionVigLabel = tk.Label(master, text='Frequency of consumption of  vegetables').grid(row=8, column=0, columnspan=3,
                                                                                                      pady=10, sticky=tk.EW)
            consumptionVig = ttk.Combobox(master, values=("Always", "Sometimes", "Rarely"))
            consumptionVig.grid(row=9, column=0, columnspan=3,pady=10)
            
            intakenLabel = tk.Label(master, text='Amount of  fluids per day').grid(row=8, column=4, columnspan=3, pady=10, sticky=tk.EW)
            cH2O = ttk.Combobox(master, values=("Less than one liter", "Between 1 and 2 liters", "More than 2 liters"))
            cH2O.grid(row=9, column=4, columnspan=3, pady=10)
            
            activityLabel = tk.Label(master, text='Frequency of alcohol consumption ').grid(row=8, column=8, columnspan=3, pady=10,
                                                                                           sticky=tk.EW)
            alkool = ttk.Combobox(master, values=("No consume alcohol", "Rarely", "Weekly", "Daily"))
            alkool.grid(row=9, column=8,columnspan=3, pady=10)
            
            # fourth row
            
            mealsLabel = tk.Label(master, text='Number of ma in meals').grid(row=10, column=0, columnspan=3, pady=10, sticky=tk.EW)
            meals = ttk.Combobox(master, values=("1", "2", "3", "more than 3"))
            meals.grid(row=11, column=0, columnspan=4, pady=10)
            
            CaloriesLabel = tk.Label(master, text='Look at the amount of calories per day ').grid(row=10, column=4, columnspan=3,
                                                                                                pady=10, sticky=tk.EW)
            Calories = ttk.Combobox(master, values=("Yes", "No"))
            Calories.grid(row=11, column=4, columnspan=3, pady=10)
            
            activityLabel = tk.Label(master, text='Type of transportation  used  ').grid(row=10, column=8, columnspan=3, pady=10,
                                                                                      sticky=tk.EW)
            transport = ttk.Combobox(master, values=("Public Transportation", "Motorbike", "Bike", "Walking", "Automobile"))
            transport.grid(row=11, column=8, columnspan=3, pady=10)
            
            text1Label = tk.Label(master, text='prediction').grid(row=12, column=0, columnspan=2, sticky=tk.EW, pady=5)
            
            #activityLabel = tk.Label(master, text='Muscle mass index').grid(row=13, column=0, columnspan=3, pady=10, sticky=tk.EW)
            MMI = tk.Entry(master, width=15)
            MMI.grid(row=14, column=0, padx=50, columnspan=3, sticky=tk.W)
            activityLabel = tk.Label(master, text='Obesity level').grid(row=13, column=4, columnspan=3, pady=10, sticky=tk.EW)
            obesityLevel = tk.Entry(master, width=15)
            obesityLevel.grid(row=14, column=4, columnspan=3, sticky=tk.W)
            processBtn = tk.Button(master, text="Process",command = lambda: Thread(target=run).start()).grid(row=14, column=7, columnspan=3, sticky=tk.W)
            homeBtn = tk.Button(text="Home").grid(row=14, column=10, columnspan=3, sticky=tk.W)
            exitBtn = ttk.Button(text="Exit", command=master.destroy).grid(row=14, column=11, columnspan=3, sticky=tk.W)

            def run():
                
                
              try:
                   ageInput = -1
                   weightInput = -1
                   heightInput = -1
                   if(int(age.get())<=20):
                       ageInput = 1
                   elif(int(age.get())>20 and int(age.get())<=40):
                       ageInput = 2
                   elif(int(age.get())>40 and int(age.get())<=60):
                       ageInput = 3
                   else:
                       ageInput = 4
                       
                   if(int(weightEntry.get())<40):
                       weightInput = 1
                   elif(int(weightEntry.get())>=40 and int(weightEntry.get())<60):
                       weightInput = 2
                   elif(int(weightEntry.get())>=60 and int(weightEntry.get())<85):
                       weightInput = 3
                   else:
                       weightInput = 4
                       
                   if(float(heightEntry.get())<1.5):
                       heightInput = 1
                   elif(float(heightEntry.get())>=1.5 and float(heightEntry.get())<1.7):
                       heightInput = 2
                   elif(float(heightEntry.get())>=1.7 and float(heightEntry.get())<1.9):
                       heightInput = 3
                   else:
                       heightInput = 4
                 
                   userInputs = np.array([float(gender.get()),ageInput,heightInput,weightInput,float(familyInput[family.get()])
                   ,float(fastFoodInput[fastFood.get()]),int(fcvc[consumptionVig.get()]),int(mealsInput[meals.get()]),float(caec[caecIntaken.get()]),
                   float(smokeInput[smoke.get()]),int(CH2O[cH2O.get()]),float(SCC[Calories.get()]),int(Physica[physical.get()]),
                   int(Technologia[activity.get()]),float(Alkool[alkool.get()]),float(MTrans[transport.get()]) ]).reshape(1,-1)
    
                  
                   X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=42) #splits data for training and predicting
                   clf = tree.DecisionTreeClassifier() #Creates classifier
                   clf = clf.fit(X_train,y_train) #Fits data into classifier
                   y_pred = clf.predict(X_test) #Predict test data
                   u_pred = clf.predict(userInputs) #Predict test data
                   print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Prints the accuracy score.
                
                   
                   output = ""
                   if ' '.join(map(str,u_pred)) == "1":
                       output = "Result : Normal Weight. This value has been calculated with {} accuracy rate".format(str(metrics.accuracy_score(y_test, y_pred)))
                   elif ' '.join(map(str,u_pred)) == "2":
                       output = "Result : Overweight Level I. This value has been calculated with {} accuracy rate".format(str(metrics.accuracy_score(y_test, y_pred)))
                   elif ' '.join(map(str,u_pred)) == "3":
                       output = "Result : Overweight Level II. This value has been calculated with {} accuracy rate".format(str(metrics.accuracy_score(y_test, y_pred)))
                   elif ' '.join(map(str,u_pred)) == "4":
                       output = "Result : Obesity Type 1. This value has been calculated with {} accuracy rate".format(str(metrics.accuracy_score(y_test, y_pred)))
                   elif ' '.join(map(str,u_pred)) == "5":
                       output = "Result : Obesity Type 2. This value has been calculated with {} accuracy rate".format(str(metrics.accuracy_score(y_test, y_pred)))
                   else:
                       output = "Result : Unpredictable Values"
                   tk.Label(master, text=output).grid(row=13, column=0, columnspan=3, pady=10, sticky=tk.EW)
        
              except KeyError:
                 print("Please fill all of the boxes")
                 
              except ValueError:
                  print("Please dont use the illegal characters like * - / ")

            tk.mainloop()
          
            
            
            
            
          
           
           
              
    













 