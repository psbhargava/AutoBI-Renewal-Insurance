import json
import plotly
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, request

import joblib
model = joblib.load('Support Vector Machine with Smote')

app = Flask(__name__)
df = pd.read_csv('./static/data_clean.csv')

def category_plot(
    cat_plot = 'histplot',
    cat_x = 'EmploymentStatus', cat_y = 'Income',
    estimator = 'count', hue = 'Response'):
    # if the selected menu is a histogram
    if cat_plot == 'histplot':
        # prepare an empty list to hold the hist configuration
        data = []
        # generate config histogram by setting the x-axis and y-axis
        for val in df[hue].unique():
            hist = go.Histogram(
                x=df[df[hue]==val][cat_x],
                y=df[df[hue]==val][cat_y],
                histfunc=estimator,
                name=val
            )
            # insert into array
            data.append(hist)
        # determine the title of the plot to be displayed
        title='Histogram'
    elif cat_plot == 'boxplot':
        data = []
        for val in df[hue].unique():
            box = go.Box(
                x=df[df[hue] == val][cat_x], # series
                y=df[df[hue] == val][cat_y], # series
                name = val
            )
            data.append(box)
        title='Box'
     # prepare the config layout where the plot will be displayed defines the name of the x-axis and y-axis   
    if cat_plot == 'histplot':
        layout = go.Layout(
            title=title,
            xaxis=dict(title=cat_x),
            yaxis=dict(title='person'),
            # boxmod group is used to group boxes based on hue
            boxmode = 'group'
        )
    else:
        layout = go.Layout(
            title=title,
            xaxis=dict(title=cat_x),
            yaxis=dict(title=cat_y),
            # boxmod group is used to group boxes based on hue
            boxmode = 'group'
        )
    # save the config plot and layout in the dictionary
    result = {'data': data, 'layout': layout}

    # json.dumps will generate the plot and store the results in graphjson
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@app.route('/')
def index():
    plot = category_plot()
    list_plot = [('histplot', 'Histogram'), ('boxplot', 'Box')]
    list_x = [('EmploymentStatus', 'EmploymentStatus'), ('State', 'State'), ('Coverage', 'Coverage'), ('Education', 'Education'), 
              ('Gender', 'Gender'), ('Vehicle_Size', 'Vehicle Size'), ('Vehicle_Class', 'Vehicle Class'), ('Marital_Status', 'Marital Status')]
    
    list_y = [('Income', 'Income'), ('Customer_Lifetime_Value', 'Customer Lifetime Value'), ('Monthly_Premium_Auto', 'Monthly Premium Auto')]
    list_est = [('count', 'Count'), ('avg', 'Average'), ('max', 'Max'), ('min', 'Min')]
    list_hue = [('Response', 'Response')]

    return render_template(
        # file that will be the response from the API
        'category.html',
        # plot to display
        plot=plot,
        # the menu that will appear in the 'Plot Type' dropdown
        focus_plot='histplot',
        # the menu that will appear in the 'X axis' dropdown
        focus_x='EmploymentStatus',

        # for the Y axis it doesn't exist, later the Y dropdown menu will be disabled because in a histogram, the Y axis will show 
        # the quantity of data menu that will appear in the 'Estimator' dropdown
        focus_estimator='count',
        # menu that will appear in the 'Hue' dropdown
        focus_hue='Response',
        # list that the loop will use to create the 'Plot Type' dropdown
        drop_plot= list_plot,
        # list that the loop will use to create the 'X Axis' dropdown
        drop_x= list_x,
        # list that the loop will use to create the 'Y Axis' dropdown
        drop_y= list_y,
        # list that will be used by looping to create the 'Estimator' dropdown
        drop_estimator= list_est,
        # list that the loop will use to create the 'Hue' dropdown
        drop_hue= list_hue)

@app.route('/cat_fn/<nav>')
def cat_fn(nav):

    # when clicking the navigation menu
    if nav == 'True':
        cat_plot = 'histplot'
        cat_x = 'EmploymentStatus'
        cat_y = 'Income'
        estimator = 'avg'
        hue = 'Response'
    
    # when selecting a value from the form
    else:
        cat_plot = request.args.get('cat_plot')
        cat_x = request.args.get('cat_x')
        cat_y = request.args.get('cat_y')
        estimator = request.args.get('estimator')
        hue = request.args.get('hue')

    # From boxplot to histogram will be None
    if estimator == None:
        estimator = 'count'
    
    # When estimator == 'count', the Y-axis dropdown menu becomes disabled and returns the value None
    if cat_y == None:
        cat_y = 'Income'

    # Dropdown menu
    list_plot = [('histplot', 'Histogram'), ('boxplot', 'Box')]
    list_x = [('EmploymentStatus', 'EmploymentStatus'), ('State', 'State'), ('Coverage', 'Coverage'), ('Education', 'Education'), 
              ('Gender', 'Gender')]
    list_y = [('Income', 'Income'), ('Customer_Lifetime_Value', 'Customer Lifetime Value'), ('Monthly_Premium_Auto', 'Monthly Premium Auto'), 
              ('Number_of_Policies', 'Number of Policies')]
    list_est = [('count', 'Count'), ('avg', 'Average'), ('max', 'Max'), ('min', 'Min')]
    list_hue = [('Response', 'Response')]

    plot = category_plot(cat_plot, cat_x, cat_y, estimator, hue)
    return render_template(
        # file that will be the response from the API
        'category.html',
        # plot to display
        plot=plot,
        # the menu that will appear in the 'Plot Type' dropdown
        focus_plot=cat_plot,
        # the menu that will appear in the 'X axis' dropdown
        focus_x=cat_x,
        focus_y=cat_y,

        # menu that will appear in the 'Estimator' dropdown
        focus_estimator=estimator,
        # menu that will appear in the 'Hue' dropdown
        focus_hue=hue,
        # list that the loop will use to create the 'Plot Type' dropdown
        drop_plot= list_plot,
        # list that the loop will use to create the 'X Axis' dropdown
        drop_x= list_x,
        # list that will be used by looping to create the 'Y Axis' dropdown
        drop_y= list_y,
        # list that will be used by looping to create the 'Estimator' dropdown
        drop_estimator= list_est,
        # list that the loop will use to create the 'Hue' dropdown
        drop_hue= list_hue
    )

# scatter plot function
def scatter_plot(cat_x, cat_y, hue):

    data = []
    for val in df[hue].unique():
        scatt = go.Scatter(
            x = df[df[hue] == val][cat_x],
            y = df[df[hue] == val][cat_y],
            mode = 'markers',
            name = str(val)
        )
        data.append(scatt)

    layout = go.Layout(
        title= 'Scatter',
        title_x= 0.5,
        xaxis=dict(title=cat_x),
        yaxis=dict(title=cat_y)
    )
    # save the config plot and layout in the dictionary
    result = {'data': data, 'layout': layout}

    # json.dumps will generate the plot and store the results in graphjson
    graphJSON = json.dumps(result,cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/scatt_fn')
def scatt_fn():
    cat_x = request.args.get('cat_x')
    cat_y = request.args.get('cat_y')
    hue = request.args.get('hue')

    # MUST! default value when scatter is first called
    if cat_x == None and cat_y == None and hue == None:
        cat_x = 'Income'
        cat_y = 'Customer_Lifetime_Value'
        hue = 'Response'

    # Dropdown menu
    list_x = [('Income', 'Income'), ('Customer_Lifetime_Value', 'Customer Lifetime Value'), ('Monthly_Premium_Auto', 'Monthly Premium Auto')]
    list_y = [('Income', 'Income'), ('Customer_Lifetime_Value', 'Customer Lifetime Value'), ('Monthly_Premium_Auto', 'Monthly Premium Auto')]
    list_hue = [('Response', 'Response')]

    plot = scatter_plot(cat_x, cat_y, hue)

    return render_template(
        'scatter.html',
        plot=plot,
        focus_x=cat_x,
        focus_y=cat_y,
        focus_hue=hue,
        drop_x= list_x,
        drop_y= list_y,
        drop_hue= list_hue
    )


def pie_plot(df, hue = 'Response'):
    vcounts = df[hue].value_counts().reset_index()
    vcounts.columns = ['labels', 'values']

    trace = go.Pie(labels=vcounts['labels'], values=vcounts['values'])
    layout = go.Layout(title='Pie')

    # save the config plot and layout in the dictionary
    result = go.Figure(data=[trace], layout=layout)

    # json.dumps will generate the plot and store the results in graphjson
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@app.route('/pie_fn')
def pie_fn():
    hue = request.args.get('hue', 'Response')
    list_hue = [('Response','Response'), ('EmploymentStatus','EmploymentStatus'), ('Coverage', 'Coverage'), 
                           ('Vehicle_Class', 'Vehicle Class')]
    
    valid_hue = [option[0] for option in list_hue]
    if hue not in valid_hue:
        hue = 'Response'
        
    plot = pie_plot(df, hue)
    return render_template('pie.html', plot=plot, focus_hue=hue, drop_hue= list_hue)


@app.route('/predict')        
def predict():
    ml=pd.read_csv('./static/data_clean_2.csv')
    tab=ml.head(1000)
    return render_template('predict.html',data=tab.values) 
            
        
@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        ## For Predict
        got=request.form
        clv =float(got['Customer Lifetime Value'])
        income =int(got['Income'])
        premi =int(got['Monthly Premium Auto'])
        claim =float(got['Total Claim Amount'])
        num_policy =int(got['Number of Policies'])

        Gender= ''
        if got['Gender'] == 'M':
            Gender= 0
        else :
            Gender=1

        Education= ''
        if got['Education'] == 'High School or Below':
            Education= 0
        elif got['Education'] == 'College':
            Education =1
        elif got['Education'] == 'Bachelor':
            Education =2
        elif got['Education'] == 'Master':
            Education =3
        else :
            Education = 4
            
        Coverage= ''
        if got['Coverage'] == 'Basic':
            Coverage = 0
        elif got['Coverage'] == 'Extended':
            Coverage = 1
        else :
            Coverage = 2
        
        Vehicle_Size=''
        if got['Vehicle Size'] == 'Small':
            Vehicle_Size= 0
        elif got['Vehicle Size'] == 'Medsize': 
            Vehicle_Size = 1
        else :
            Vehicle_Size = 2
        
        Vehicle_Class_2door=''  
        if got['Vehicle Class'] == 'Two-Door Car':
            Vehicle_Class_2door = 1
        else :
            Vehicle_Class_2door = 0

        Vehicle_Class_4door=''  
        if got['Vehicle Class'] == 'Four-Door Car':
            Vehicle_Class_4door = 1
        else :
            Vehicle_Class_4door = 0

        Vehicle_Class_SUV=''  
        if got['Vehicle Class'] == 'SUV':
            Vehicle_Class_SUV = 1
        else :
            Vehicle_Class_SUV = 0

        Vehicle_Class_Luxury_SUV=''  
        if got['Vehicle Class'] == 'Luxury SUV':
            Vehicle_Class_Luxury_SUV = 1
        else :
            Vehicle_Class_Luxury_SUV = 0

        Vehicle_Class_Luxury=''  
        if got['Vehicle Class'] == 'Luxury Car':
            Vehicle_Class_Luxury = 1
        else :
            Vehicle_Class_Luxury = 0

        Vehicle_Class_Sports=''  
        if got['Vehicle Class'] == 'Sports Car':
            Vehicle_Class_Sports = 1
        else :
            Vehicle_Class_Sports = 0

        State_Arizona=''  
        if got['State'] == 'Arizona':
            State_Arizona = 1
        else :
            State_Arizona = 0
            
        State_California=''  
        if got['State'] == 'California':
            State_California = 1
        else :
            State_California = 0

        State_Nevada=''  
        if got['State'] == 'Nevada':
            State_Nevada = 1
        else :
            State_Nevada = 0
            
        State_Oregon=''  
        if got['State'] == 'Oregon':
            State_Oregon = 1
        else :
            State_Oregon = 0

        State_Washington=''  
        if got['State'] == 'Washington':
            State_Washington = 1
        else :
            State_Washington = 0

        Employment_employed=''  
        if got['EmploymentStatus'] == 'Employed':
            Employment_employed = 1
        else :
            Employment_employed = 0

        Employment_unemployed=''  
        if got['EmploymentStatus'] == 'Unemployed':
            Employment_unemployed = 1
        else :
            Employment_unemployed = 0

        Employment_Medical_Leave=''  
        if got['EmploymentStatus'] == 'Medical Leave':
            Employment_Medical_Leave = 1
        else :
            Employment_Medical_Leave = 0

        Employment_Disabled=''  
        if got['EmploymentStatus'] == 'Disabled':
            Employment_Disabled = 1
        else :
            Employment_Disabled = 0

        Employment_Retired=''  
        if got['EmploymentStatus'] == 'Retired':
            Employment_Retired = 1
        else :
            Employment_Retired = 0

        location_suburb=''  
        if got['Location Code'] == 'Suburban':
            location_suburb = 1
        else :
            location_suburb = 0
        
        location_urb=''  
        if got['Location Code'] == 'Urban':
            location_urb = 1
        else :
            location_urb = 0
            
        location_rural=''  
        if got['Location Code'] == 'Rural':
            location_rural = 1
        else :
            location_rural = 0
        
        marital_married=''  
        if got['Marital Status'] == 'Married':
            marital_married = 1
        else :
            marital_married = 0

        marital_single=''  
        if got['Marital Status'] == 'Single':
            marital_single = 1
        else :
            marital_single = 0
        
        marital_divorced=''  
        if got['Marital Status'] == 'Divorced':
            marital_divorced = 1
        else :
            marital_divorced = 0
        
        Policy_personal1=''  
        if got['Policy'] == 'Personal L1':
            Policy_personal1 = 1
        else :
            Policy_personal1 = 0

        Policy_personal2=''  
        if got['Policy'] == 'Personal L2':
            Policy_personal2 = 1
        else :
            Policy_personal2 = 0

        Policy_personal3=''  
        if got['Policy'] == 'Personal L3':
            Policy_personal3 = 1
        else :
            Policy_personal3 = 0
        
        Policy_corporate1=''  
        if got['Policy'] == 'Corporate L1':
            Policy_corporate1 = 1
        else :
            Policy_corporate1 = 0

        Policy_corporate2=''  
        if got['Policy'] == 'Corporate L2':
            Policy_corporate2 = 1
        else :
            Policy_corporate2 = 0

        Policy_corporate3=''  
        if got['Policy'] == 'Corporate L3':
            Policy_corporate3 = 1
        else :
            Policy_corporate3 = 0

        Policy_special1=''  
        if got['Policy'] == 'Special L1':
            Policy_special1 = 1
        else :
            Policy_special1 = 0

        Policy_special2=''  
        if got['Policy'] == 'Special L2':
            Policy_special2 = 1
        else :
            Policy_special2 = 0
        
        Policy_special3=''  
        if got['Policy'] == 'Special L3':
            Policy_special3 = 1
        else :
            Policy_special3 = 0

        offer_1=''  
        if got['Renew Offer Type'] == 'Offer1':
            offer_1 = 1
        else :
            offer_1 = 0

        offer_2=''  
        if got['Renew Offer Type'] == 'Offer2':
            offer_2 = 1
        else :
            offer_2 = 0

        offer_3=''  
        if got['Renew Offer Type'] == 'Offer3':
            offer_3 = 1
        else :
            offer_3 = 0
        
        offer_4=''  
        if got['Renew Offer Type'] == 'Offer4':
            offer_4 = 1
        else :
            offer_4 = 0

        channel_agent=''  
        if got['Sales Channel'] == 'Agent':
            channel_agent = 1
        else :
            channel_agent = 0

        channel_branch=''  
        if got['Sales Channel'] == 'Branch':
            channel_branch = 1
        else :
            channel_branch = 0
            
        channel_web=''  
        if got['Sales Channel'] == 'Web':
            channel_web = 1
        else :
            channel_web = 0

        channel_call=''  
        if got['Sales Channel'] == 'Call Center':
            channel_call = 1
        else :
            channel_call = 0

        pred = ((model.predict_proba([[clv, income, premi, claim, num_policy, Gender, Education, Coverage, Vehicle_Size, 
                                       Vehicle_Class_2door, Vehicle_Class_4door, Vehicle_Class_SUV, Vehicle_Class_Luxury, 
                                       Vehicle_Class_Luxury_SUV, Vehicle_Class_Sports, State_Arizona, State_California, 
                                       State_Nevada, State_Oregon, State_Washington, Employment_employed, Employment_unemployed, 
                                       Employment_Medical_Leave, Employment_Disabled, Employment_Retired, location_suburb, location_urb, 
                                       location_rural, marital_married, marital_single, marital_divorced, Policy_personal1, Policy_personal2, 
                                       Policy_personal3, Policy_corporate1, Policy_corporate2, Policy_corporate3, Policy_special1, 
                                       Policy_special2, Policy_special3, offer_1, offer_2, offer_3, offer_4, channel_agent, channel_branch,
                                       channel_web, channel_call]])[0][1])*100).round(2)
        ### For Fill Data
        v_class_dt= ''
        if got['Vehicle Class'] == 'Two-Door Car':
            v_class_dt= 'Two-Door Car'

        elif got['Vehicle Class'] == 'Four-Door Car':
            v_class_dt='Four-Door Car'
            
        elif got['Vehicle Class'] == 'SUV':
            v_class_dt='SUV'
            
        elif got['Vehicle Class'] == 'Luxury SUV':
            v_class_dt='Luxury SUV'

        else:
            v_class_dt= 'Sports Car'
            
            
        state_dt= ''
        if got['State'] == 'Arizona':
            state_dt='Arizona'
            
        elif got['State'] == 'California':
            state_dt='California'
            
        elif got['State'] == 'Nevada':
            state_dt='Nevada'

        elif got['State'] == 'Oregon':
            state_dt='Oregon'
        
        else :
            state_dt = 'Washington'
        

        employment_dt= ''
        if got['EmploymentStatus'] == 'Employed':
            employment_dt='Employed'
            
        elif got['EmploymentStatus'] == 'Unemployed':
            employment_dt='Unemployed'
            
        elif got['EmploymentStatus'] == 'Medical Leave':
            employment_dt='Medical Leave'

        elif got['EmploymentStatus'] == 'Disabled':
            employment_dt='Disabled'
        
        else :
            employment_dt = 'Retired'
            

        location_dt= ''
        if got['Location Code'] == 'Suburban':
            location_dt='Suburban'
            
        elif got['Location Code'] == 'Urban':
            location_dt='Urban'
        
        else :
            location_dt = 'Rural'

        marital_dt= ''
        if got['Marital Status'] == 'Married':
            marital_dt='Married'
            
        elif got['Marital Status'] == 'Single':
            marital_dt='Single'
        
        else :
            marital_dt = 'Divorced'

        policy_dt= ''
        if got['Policy'] == 'Personal L1':
            policy_dt='Personal L1'
            
        elif got['Policy'] == 'Personal L2':
            policy_dt='Personal L2'

        elif got['Policy'] == 'Personal L3':
            policy_dt='Personal L3'
        
        elif got['Policy'] == 'Corporate L1':
            policy_dt='Corporate L1'

        elif got['Policy'] == 'Corporate L2':
            policy_dt='Corporate L2'

        elif got['Policy'] == 'Corporate L3':
            policy_dt='Corporate L3'

        elif got['Policy'] == 'Special L1':
            policy_dt='Special L1'

        elif got['Policy'] == 'Special L2':
            policy_dt='Special L2'

        else :
            policy_dt = 'Special L3'

        offer_dt= ''
        if got['Renew Offer Type'] == 'Offer1':
            offer_dt='Offer1'
            
        elif got['Renew Offer Type'] == 'Offer2':
            offer_dt='Offer2'

        elif got['Renew Offer Type'] == 'Offer3':
            offer_dt='Offer3'
        else :
            offer_dt = 'Offer4'

        channel_dt= ''
        if got['Sales Channel'] == 'Agent':
            channel_dt='Agent'
            
        elif got['Sales Channel'] == 'Branch':
            channel_dt='Branch'

        elif got['Sales Channel'] == 'Web':
            channel_dt='Web'
            
        else :
            channel_dt = 'Call Center'
        
        gender_dt = ''
        if got['Gender'] == 'M':
            gender_dt = 'Male'
        else:
            gender_dt = 'Female'

        education_dt= ''
        if got['Education'] == 'High School or Below':
            education_dt= 'High School or Below'
        elif got['Education'] == 'College':
            education_dt = 'College'
        elif got['Education'] == 'Bachelor':
            education_dt = 'Bachelor'
        elif got['Education'] == 'Master':
            education_dt = 'Master'
        else :
            education_dt = 'Doctor'

        coverage_dt= ''
        if got['Coverage'] == 'Basic':
            coverage_dt = 'Basic'
        elif got['Coverage'] == 'Extended':
            coverage_dt = 'Extended'
        else :
            coverage_dt = 'Premium'
        
        v_size_dt=''
        if got['Vehicle Size'] == 'Small':
            v_size_dt= 'Small'
        elif got['Vehicle Size'] == 'Medsize': 
            v_size_dt = 'Medsize'
        else :
            v_size_dt = 'Large'

        return render_template('result.html', clv =float(got['Customer Lifetime Value']), income =int(got['Income']), 
                               premi =int(got['Monthly Premium Auto']), claim =float(got['Total Claim Amount']), 
                               num_policy =int(got['Number of Policies']), Gender= gender_dt, Education = education_dt,
                               Coverage = coverage_dt, Vehicle_Size = v_size_dt, Vehicle_Class = v_class_dt, State = state_dt,
                               Employment = employment_dt, Location = location_dt, Marital = marital_dt, Policy = policy_dt,
                               Offer = offer_dt, Channel = channel_dt, pred=pred)



if __name__ == "__main__":
    app.run(debug=True, port=5000, host="localhost", use_reloader=True)

