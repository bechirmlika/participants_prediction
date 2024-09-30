import pandas as pd 
import json
import boto3
import numpy as np
import io
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

def predictionBasedOnCom(event, context):

    # Importing data
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket='lambda.data.events', Key='lambda_data_events.csv')
    contents = data['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(contents))
    
    # Preprocessing steps
    df.dropna(inplace=True)
    
    eventData = event['eventData']
    startDate = eventData["startDate"]
    startTime = datetime.datetime.strptime(startDate, "%Y-%m-%dT%H:%M:%S.%fZ").hour
    title = eventData["title"]
    title_length = len(title)
    title_length_squared = title_length ** 2
    description = eventData["description"]
    description_length = len(description)
    description_length_squared = description_length ** 2
    address = eventData["address"]
    access = eventData["access"]
    category = eventData["category"]
    type = eventData["type"]
    isFree = eventData["isFree"]
    nbrOfRegistrationFields = event["nbrOfRegistrationFields"]
    requiredSignIn = eventData["requiredSignIn"]
    speakers = eventData['speakers']
    nbrOfSpeakers = len(speakers)
    speakerNbrOfEvents = event["speakerNbrOfEvents"]
    speakerNbrOfParticipants = event["speakerNbrOfParticipants"]
    sponsors = eventData["sponsors"]
    nbrOfSponsors = len(sponsors)
    sponsorNbrOfEvents = event["sponsorNbrOfEvents"]
    sponsorNbrOfParticipants = event["sponsorNbrOfParticipants"]
    SourceDirect = event["SourceDirect"]
    SourceSocial = event["SourceSocial"]
    
    # Automated encoding using LabelEncoder
    le = LabelEncoder()
    df['access'] = le.fit_transform(df['access'])
    df['category'] = le.fit_transform(df['category'])
    df['type'] = le.fit_transform(df['type'])
    df['isFree'] = le.fit_transform(df['isFree'])
    df['requiredSignIn'] = le.fit_transform(df['requiredSignIn'])
    
    # Feature selection and scaling
    features = df.drop(['event_registered', 'event_visitors', 'event_conversionRate'], axis=1)
    label = df['event_registered']
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # New event data for prediction
    Xnew = [[startTime, title_length, title_length_squared, address, access, category, type, isFree, requiredSignIn,
             description_length_squared, nbrOfRegistrationFields, nbrOfSpeakers, speakerNbrOfEvents, speakerNbrOfParticipants, 
             nbrOfSponsors, sponsorNbrOfEvents, sponsorNbrOfParticipants, SourceDirect, SourceSocial]]

    # Models to evaluate
    models = {
        'RandomForest': RandomForestRegressor(),
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor()
    }
    
    best_model = None
    best_score = -1
    
    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        print(f"{model_name} Test Score: {score:.2%}")
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Ensure we have a valid trained model
    if best_model is not None:
        # Make multiple predictions with the best model to show a range
        predictions = []
        
        for _ in range(3):  # Predict 3 times
            ynew = best_model.predict(Xnew)
            predictions.append(ynew[0])

        # Calculate the min and max predictions
        min_prediction = max(5, min(predictions))  
        max_prediction = max(5, max(predictions))  

        result = {
            "min_prediction": min_prediction,
            "max_prediction": max_prediction
        }

        print(f"Best model: {best_model.__class__.__name__}")
        print(f"Predicted Registered Range: Min={min_prediction}, Max={max_prediction}")
    else:
        result = {
            "error": "No model could be trained successfully."
        }

    return result