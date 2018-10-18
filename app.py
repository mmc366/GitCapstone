from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

#Plotting imports
#from __future__ import print_function
from math import pi, sin, cos
from bokeh.embed import components
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.util.browser import view
from bokeh.models.glyphs import Circle, Arc, Ray, Text
from bokeh.models import ColumnDataSource, Range1d, Plot

#server = flask.Flask(__name__)
#app = dash.Dash(__name__, server=server)

app = Flask(__name__)

model1 = joblib.load('logitmodel2.joblib') 

def transform(user_input):
    """
    Computes dummy coding on categorical predictors and calculations for
    Alcohol Consumption and BMI from user survey input. Returns a list
    of features to be passed to logistic regression model.
    
    @user_input: list<int>
    @return: DataFrame (was np.array<int>) 
    """
   
    #Alcohol Consumption:
    
    #Calculates Alcohol Consumption and adds to user_input df
    user_input['Alcohol Consumption'] = user_input['Alcohol Days']*user_input['Alcohol Drinks']
    
    #Drops Alcohol Days and Alcohol Drinks columns
    user_input = user_input.drop(columns=['Alcohol Days', 'Alcohol Drinks'])
    
    #BMI:
    
    #Converting lbs to kgs
    wtkg = float(user_input['Weight'])*0.453592
    
    #Converting inches to meters
    htinches = user_input['Height_feet']*12 + user_input['Height_inches']
    htmeter = htinches*0.0254
    
    #Calculating numeric BMI
    bmi_numeric = (wtkg/htmeter)/htmeter
    bmi_numeric = bmi_numeric.iloc[0]
    
    #Categorizing BMI
    def categorize_BMI(bmi_numeric):
        """
        Takes the numeric BMI calculation and reutrns corresponding
        category label.
        
        @bmi_numeric = int
        @return = int (1 = Underweight, 2 = Normal Weight, 3 = Overweight, 4 = Obese)
        """
        if bmi_numeric < 18.50:
            return 1
        if bmi_numeric >= 18.50 and bmi_numeric < 25.00:
            return 2
        if bmi_numeric >= 25.00 and bmi_numeric < 30.00:
            return 3
        if bmi_numeric >= 30.00:
            return 4
    
    #Adds BMI category to user_input df
    user_input['BMI'] = categorize_BMI(bmi_numeric)
    
    #Drops Weight, Height_inches, and Height_feet columns
    user_input = user_input.drop(columns=['Weight', 'Height_inches', 'Height_feet'])
    
    #Removed 3 predictors w/ most missing data
    no_sugar_marijuana = user_input.drop(columns=['Soda Consumption', 'Sugary Drink Consumption', 
                                                  'Marijuana Use', 'Life Satisfaction', 'Calorie Informed Choices'])
    
    user_input_final = no_sugar_marijuana[['Average Sleep', 'Sex', 'Marital', 'Employment', 'Physical QoL',
       'Mental QoL', 'Physical Activity', 'Race', 'Age', 'BMI', 'Education',
       'Income', 'Smoker Status', 'Alcohol Consumption',
       'HeartAttackOutcomeDummy', 'AnginaCoronaryHDOutcomeDummy',
       'StrokeOutcomeDummy', 'AsthmaLifetimeOutcomeDummy',
       'SkinCancerOutcomeDummy', 'OtherCancerOutcomeDummy',
       'COPDEmphysemaChronicBroncOutcomeDummy', 'ArthritisOutcomeDummy',
       'DepressionOutcomeDummy', 'KidneyDiseaseOutcomeDummy']]
    
    return user_input_final #user_input_final.values

#def transform_test():
    #Tests transform()
    test_df = pd.DataFrame(data=[[0, 2, 1, 1, 2, 3, 3, 20, 15, 0, 
                                             5, 3, 1, 20, 8, 150, 5, 7, 1, 0, 
                                             0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                                             1]], columns=['Sex', 'Age',
                                  'Race', 'Marital', 'Employment', 'Income', 'Education', 'Soda Consumption', 
                                  'Sugary Drink Consumption', 'Physical Acvivity', 'Alcohol Days', 'Alcohol Drinks',
                                  'Smoker Status', 'Marijuana Use', 'Average Sleep', 'Weight', 'Height_feet',
                                  'Height_inches', 'Calorie Informed Choices', 'HeartAttackOutcomeDummy',
                                  'AnginaCoronaryHDOutcomeDummy', 'StrokeOutcomeDummy', 'AsthmaLifetimeOutcomeDummy',
                                  'SkinCancerOutcomeDummy', 'OtherCancerOutcomeDummy', 'COPDEmphysemaChronicBroncOutcomeDummy',
                                  'ArthritisOutcomeDummy', 'DepressionOutcomeDummy', 'KidneyDiseaseOutcomeDummy',
                                  'Life Satisfaction', 'Physical QoL', 'Mental QoL'])
    
    #assert (transform_test(test_df)).values == np.array([[0, 2, 1, 1, 2, 3, 3, 20, 15, 0, 
    #                                         5, 3, 1, 20, 8, 150, 5, 7, 1, 0, 
    #                                         0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
    #                                         1]])

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/survey', methods=["GET", "POST"])
def survey():
    if request.method == "GET":
        return render_template('survey.html')
 
    else: #Request was a POST
        
        user_input = pd.DataFrame(data=[[request.form['Average_Sleep'],
                                         request.form['Sex'],
                                         request.form['Marital'],
                                         request.form['Employment'],
                                         request.form['Physical_QoL'],
                                         request.form['Mental_QoL'],
                                         request.form['Physical_Activity'],
                                         request.form['Race'],
                                         request.form['Age'],
                                         request.form['Weight'], 
                                         request.form['Height_feet'],
                                         request.form['Height_inches'],                                           
                                         request.form['Education'],
                                         request.form['Income'],
                                         request.form['Soda_Consumption'], 
                                         request.form['Sugary_Drink_Consumption'],                                          
                                         request.form['ALCDAY5'], 
                                         request.form['AVEDRNK2'],
                                         request.form['Smoker_Status'], 
                                         request.form['Marijuana_Use'],   
                                         request.form['Calorie_Informed_Choices'], 
                                         request.form['HeartAttackOutcomeDummy'],
                                         request.form['AnginaCoronaryHDOutcomeDummy'], 
                                         request.form['StrokeOutcomeDummy'],
                                         request.form['AsthmaLifetimeOutcomeDummy'], 
                                         request.form['SkinCancerOutcomeDummy'],
                                         request.form['OtherCancerOutcomeDummy'], 
                                         request.form['COPDEmphysemaChronicBroncOutcomeDummy'],
                                         request.form['ArthritisOutcomeDummy'], 
                                         request.form['DepressionOutcomeDummy'], 
                                         request.form['KidneyDiseaseOutcomeDummy'], 
                                         request.form['Life_Satisfaction']]],                        
                    columns=['Average Sleep', 'Sex', 'Marital', 'Employment', 
                             'Physical QoL', 'Mental QoL', 'Physical Activity',
                             'Race', 'Age', 'Weight', 'Height_feet','Height_inches',
                             'Education', 'Income', 'Soda Consumption', 
                             'Sugary Drink Consumption', 
                             'Alcohol Days', 'Alcohol Drinks','Smoker Status',
                             'Marijuana Use', 'Calorie Informed Choices', 
                             'HeartAttackOutcomeDummy', 'AnginaCoronaryHDOutcomeDummy', 
                             'StrokeOutcomeDummy', 'AsthmaLifetimeOutcomeDummy',
                             'SkinCancerOutcomeDummy', 'OtherCancerOutcomeDummy', 
                             'COPDEmphysemaChronicBroncOutcomeDummy',
                             'ArthritisOutcomeDummy', 'DepressionOutcomeDummy', 
                             'KidneyDiseaseOutcomeDummy', 'Life Satisfaction'], dtype=np.float)#.astype(float)
        
        
        #return model1.predict(transform(user_input))
       # print(user_input.values)
        prediction=model1.predict_proba(transform(user_input).values)[-1][-1]
        
        def risk_category(risk_probability):
            """
            Takes a risk probability (e.g., "prediction") as a float and returns
            the corresponding risk level category as a string.
            
            @risk_probably: float
            @reutnr: str
            """
            if risk_probability < 0.25:
                return "Low Risk"
            if risk_probability >= 0.25 and risk_probability < 0.50:
                return "Low Moderate Risk"
            if risk_probability >=.50 and risk_probability < 0.75:
                return "High Moderate Risk"
            else:
                return "High Risk"
            
        def adjusted_risk(user_input): #Need to test this function
            """
            Takes DataFrame of user_input, transforms it, and calculates
            adjusted predicted risk based on changes to modifible health
            behaviors. Returns a new adjusted probability (float) and
            the list of suggested health behavior changes contributing
            to the reduced risk.
            
            @user_input: DataFrame
            @return: float, list<str>
            """
            transformed_df = transform(user_input)
            
            suggested_modifications = [] #Append to suggested modification list
            
            #Sets sleep to average sleep recommended for adults 18 - 64 yrs 
            #(7-9 hrs), adults 65+ (7-8 hrs) by National Sleep Foundation
            if transformed_df['Average Sleep'].values[0] < 7.0 or transformed_df['Average Sleep'].values[0] > 9.0:
                orig_sleep = transformed_df['Average Sleep'].values[0]
                transformed_df['Average Sleep'].values[0] = 8.0
                suggested_modifications.append("Get an average of 8.0 hours of sleep per night instead of "+ str(orig_sleep)+ " hours.")
            
            #Sets Physical Activity to had physical activity in past 30 days
            if transformed_df['Physical Activity'].values[0] == 0:
                transformed_df['Physical Activity'].values[0] = 1
                suggested_modifications.append("Incorporate a regular exercise routine into your schedule – this could be as simple as walking versus driving to the corner store or  taking the stairs instead of the elevator.")
            
            #Sets Obese -> Overweight and Overweight -> Normal weight
            if transformed_df['BMI'].values[0] == 3.0 or transformed_df['BMI'].values[0] == 4.0:
                orig_BMI = transformed_df['BMI'].values[0]
                transformed_df['BMI'].values[0] = transformed_df['BMI'].values[0] - 1.0
                
                if orig_BMI == 3.0:
                    orig_BMI = "Overweight (BMI: 25 –  < 30)"
                    suggested_BMI = "Normal Weight (BMI: 18.5 - < 25)"
                    suggested_modifications.append("Reduce BMI through gradual, healthy weight loss from "+ orig_BMI+ " to "+ suggested_BMI+ ".")
                    
                if orig_BMI == 4.0:
                    orig_BMI = "Obese (BMI: >= 30)"
                    suggested_BMI = "Overweight (BMI: 25 –  < 30)"
                    suggested_modifications.append("Reduce BMI through gradual, healthy weight loss from "+ orig_BMI+ " to "+ suggested_BMI+ ".")
            
            #Sets Underweight -> Normal weight
            if transformed_df['BMI'].values[0] == 1:
                transformed_df['BMI'].values[0] = transformed_df['BMI'].values[0] + 1.0
                orig_BMI = "Underweight (BMI: < 18.5)"
                suggested_BMI = "Normal Weight (BMI: 18.5 - < 25)"
                suggested_modifications.append("Reduce BMI through gradual, healthy weight loss from "+ orig_BMI+ " to "+ suggested_BMI+ ".")
            
            #Sets Daily smokers -> Occasional smokers and Occasional smokers -> Former smokers  
            if transformed_df['Smoker Status'].values[0] == 1.0 or transformed_df['Smoker Status'].values[0] == 2.0:
                orig_smoker = transformed_df['Smoker Status'].values[0]
                transformed_df['Smoker Status'].values[0] = transformed_df['Smoker Status'].values[0] + 1.0
                
                if orig_smoker == 1.0:
                    orig_smoker = "smoking every day"
                    suggested_smoker = "smoking some days"
                    suggested_modifications.append("Reduce smoking frequency from "+orig_smoker+" to "+suggested_smoker+ ".")
                    
                if orig_smoker == 2.0:
                    orig_smoker = "smoking some days"
                    suggested_smoker = "former smoker (quit)"
                    suggested_modifications.append("Reduce smoking frequency from "+orig_smoker+" to "+suggested_smoker+ ".")
            
            #Reduces weekly alcohol consumption by 25% (cutoff arbitrary)
            if transformed_df['Alcohol Consumption'].values[0] >= 1.0:
                orig_alcohol = transformed_df['Alcohol Consumption'].values[0]
                suggested_alcohol = transformed_df['Alcohol Consumption'].values[0]*0.25
                transformed_df['Alcohol Consumption'].values[0] = transformed_df['Alcohol Consumption'].values[0]*0.25
                suggested_modifications.append("Reduce average weekly alcohol consumption from " +str(orig_alcohol)+ " drink(s) per week to " +str(suggested_alcohol)+ " drink(s) per week.")
                
            #Sets 1-13 days poor Mental QoL -> 0 days and 14+ poor days -> 1-13 days
            if transformed_df['Mental QoL'].values[0] == 2.0 or transformed_df['Mental QoL'].values[0] == 3.0:
                #orig_mental = transformed_df['Mental QoL'].values[0]
                transformed_df['Mental QoL'].values[0] = transformed_df['Mental QoL'].values[0] - 1.0
                suggested_modifications.append("Consider consulting a psychologist/psychiatrist to learn tools for coping with stress and emotional difficulties in order to improve your mental health.")
                    
                
            return (transformed_df, suggested_modifications)
          
        adjusted_prediction=model1.predict_proba(adjusted_risk(user_input)[0].values)[-1][-1]
        
        modification_list=adjusted_risk(user_input)[1]
        
                
#Gauge Chart Rendering        
        xdr = Range1d(start=-1.25, end=1.25)
        ydr = Range1d(start=-1.25, end=1.25)
        
        plot = Plot(x_range=xdr, y_range=ydr, plot_width=500, plot_height=500)
        plot.title.text = "Predicted Diabetes Risk"
        plot.toolbar_location = None
        
        start_angle = pi + pi/4
        end_angle = -pi/4
        
        max_kmh = 1.0
        max_mph = 1.0
        
        major_step, minor_step = .25, .05
        
        plot.add_glyph(Circle(x=0, y=0, radius=1.00, fill_color="white", line_color="black"))
        plot.add_glyph(Circle(x=0, y=0, radius=0.05, fill_color="gray", line_color="black"))
        
        plot.add_glyph(Text(x=0, y=+0.15, text=["Current Risk Probability"], text_color="red", text_align="center", text_baseline="bottom", text_font_style="bold"))
        plot.add_glyph(Text(x=0, y=-0.15, text=["Adjusted Risk Probability"], text_color="blue", text_align="center", text_baseline="top", text_font_style="bold"))
        
        def data(value):
            """Shorthand to override default units with "data", for e.g. `Ray.length`. """
            return dict(value=value, units="data")
        
        def speed_to_angle(speed, units):
            max_speed = max_kmh
            speed = min(max(speed, 0), max_speed)
            total_angle = start_angle - end_angle
            angle = total_angle*float(speed)/max_speed
            return start_angle - angle
        
        def add_needle(speed, units, color_choice, line_weight):
            angle = speed_to_angle(speed, units)
            plot.add_glyph(Ray(x=0, y=0, length=data(0.75), angle=angle,    line_color=color_choice, line_width=line_weight))
            plot.add_glyph(Ray(x=0, y=0, length=data(0.10), angle=angle-pi, line_color=color_choice, line_width=line_weight))
        
        def polar_to_cartesian(r, alpha):
            return r*cos(alpha), r*sin(alpha)
        
        def add_gauge(radius, max_value, length, direction, color, major_step, minor_step):
            major_angles, minor_angles = [], []
            major_labels, minor_labels = [], []
        
            total_angle = start_angle - end_angle
        
            major_angle_step = float(major_step)/max_value*total_angle
            minor_angle_step = float(minor_step)/max_value*total_angle
        
            major_angle = 0
        
            while major_angle <= total_angle:
                major_angles.append(start_angle - major_angle)
                major_angle += major_angle_step
        
            minor_angle = 0
        
            while minor_angle <= total_angle:
                minor_angles.append(start_angle - minor_angle)
                minor_angle += minor_angle_step
        
            major_labels = [ major_step*i for i, _ in enumerate(major_angles) ]
            minor_labels = [ minor_step*i for i, _ in enumerate(minor_angles) ]
        
            n = major_step/minor_step
            minor_angles = [ x for i, x in enumerate(minor_angles) if i % n != 0 ]
            minor_labels = [ x for i, x in enumerate(minor_labels) if i % n != 0 ]
        
            glyph = Arc(x=0, y=0, radius=radius, start_angle=start_angle, end_angle=end_angle, direction="clock", line_color=color, line_width=2)
            plot.add_glyph(glyph)
        
            rotation = 0 if direction == 1 else -pi
        
            x, y = zip(*[ polar_to_cartesian(radius, angle) for angle in major_angles ])
            angles = [ angle + rotation for angle in major_angles ]
            source = ColumnDataSource(dict(x=x, y=y, angle=angles))
        
            glyph = Ray(x="x", y="y", length=data(length), angle="angle", line_color=color, line_width=2)
            plot.add_glyph(source, glyph)
        
            x, y = zip(*[ polar_to_cartesian(radius, angle) for angle in minor_angles ])
            angles = [ angle + rotation for angle in minor_angles ]
            source = ColumnDataSource(dict(x=x, y=y, angle=angles))
        
            glyph = Ray(x="x", y="y", length=data(length/2), angle="angle", line_color=color, line_width=1)
            plot.add_glyph(source, glyph)
        
            x, y = zip(*[ polar_to_cartesian(radius+2*length*direction, angle) for angle in major_angles ])
            text_angles = [ angle - pi/2 for angle in major_angles ]
            source = ColumnDataSource(dict(x=x, y=y, angle=text_angles, text=major_labels))
        
            glyph = Text(x="x", y="y", angle="angle", text="text", text_align="center", text_baseline="middle")
            plot.add_glyph(source, glyph)
        
        add_gauge(0.75, max_kmh, 0.05, +1, "red", major_step, minor_step)
        add_gauge(0.70, max_mph, 0.05, -1, "blue", major_step, minor_step)
        
        add_needle(prediction, "Current Risk", "red", 6)
        add_needle(adjusted_prediction, "Adjusted Risk", "blue", 3)
        
        script, div = components(plot)
        
        return render_template('orig_output.html', script=script, div=div, prediction=prediction, adjusted_prediction=adjusted_prediction, risk_level=risk_category(prediction), adjusted_risk_level=risk_category(adjusted_prediction), modifications=modification_list)
        #return render_template('current_risk.html', prediction)

        

if __name__ == '__main__':
  app.run(port=33507, debug=True)
