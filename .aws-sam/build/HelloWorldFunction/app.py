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

def lambda_handler(event, context):
    if "commData" in event:
        return predictionBasedOnCom(event, context)
    else:
        return predictionBasedOnComV2(event, context)  

def predictionBasedOnComV2(event, context):
    
     print("************* predictionBasedOnCom2.0 ***************")
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
     titleIsQuestion = 0.2 if title.endswith('?') else 0.4
     if 'english' in title:
        titleLanguage = 2
     elif 'french' in title:
        titleLanguage = 3
     else:
        titleLanguage = 1
     title_length = len(title)
     title_length_squared = title_length * title_length
     description = eventData["description"]
     description_length = len(description)
     description_length_squared = description_length * description_length
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
     if "SourceDirect" in event:
        SourceDirect = event["SourceDirect"]
     else:
        SourceDirect = 50
     if "SourceSocial" in event:
        SourceSocial = event["SourceSocial"]
     else:
        SourceSocial = 20
     LE = LabelEncoder()
     columns_to_encode = ['start_time', 'title_is_question', 'title_language', 'platform', 'location', 'access', 
     'category', 'type', 'isFree', 'requiredSignIn', 'user_device', 'user_os', 'user_country','Tag1','Tag2','Tag3','Tag4']
     for column in columns_to_encode:
        df[column] = LE.fit_transform(df[column])
     features = df.drop(['event_registered','event_visitors','event_conversionRate'],axis=1)
     label = df.event_registered
     X_train, X_test, y_train, y_test = train_test_split(features,label, test_size=0.2, random_state=0)
     sc = MinMaxScaler()
     features = sc.fit_transform(features)
     features = df.drop(['event_registered','event_visitors','event_conversionRate'],axis=1)
     label = df.event_registered
     X_train, X_test, y_train, y_test = train_test_split(features,label, test_size=0.2, random_state=0) 
    # Training models 
     variables_to_encode = [address, access, category, type]
     mappingType = {"virtual": 3, "physical": 1, "hybrid": 2}
     encoded_type = mappingType[type]
     if address == 'Evey virtual venue':
        address_int = 2
     else:
        address_int = 1
     mappingAccess = {"public": 3, "private": 1, "restricted": 2}
     encoded_access = mappingAccess[access]
     mappingCategory = {"event": 7, "conference": 6, "fair": 5, "training": 4, "bootcamp": 3, "show": 2, "other": 1}
     encoded_category = mappingCategory[category]
     encoded_isFree = 2 if isFree else 1
     encoded_requiredSignIn = 0.5 if requiredSignIn else 1
     if speakerNbrOfParticipants == 0:
        speakerNbrOfParticipants = 20
     Xnew = [[1, startTime, title_length, title_length_squared, titleIsQuestion, titleLanguage, description_length,description_length_squared, 2, 
     address_int, encoded_access*12, encoded_category, encoded_type, encoded_isFree, nbrOfRegistrationFields, encoded_requiredSignIn,
     nbrOfSpeakers, speakerNbrOfEvents, speakerNbrOfParticipants, nbrOfSponsors, sponsorNbrOfEvents, sponsorNbrOfParticipants*0.2, 2, 1, 1, 25, 1,
     SourceDirect, SourceSocial, 10, 10, 1, 1, 1, 1]]
     RF = RandomForestRegressor()
     RF.fit(X_train, y_train)
     y_pred = RF.predict(X_test)
     ynew = RF.predict(Xnew)
     prediction = max(5, ynew[0])
     prediction_weighted = prediction * 0.3
     # Calcul de l'intervalle de confiance
     tolerance = 0.25 * prediction_weighted
     lower_bound = max(prediction_weighted - tolerance, 5)
     upper_bound = prediction_weighted + tolerance
     result = {
        "prediction": prediction_weighted,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }
     print("Random Forest Regressor")
     print("------------------------")
     print('Score Train : ',"{:.2%}".format(RF.score(X_train, y_train)))
     print('Score Test  : ',"{:.2%}".format(RF.score(X_test, y_test)))
     print("X=%s, Predicted_Registered=%s" % (Xnew[0], ynew[0]))
     print("Intervalle de confiance : [{:.2f}, {:.2f}]".format(lower_bound, upper_bound))
     print("------------------------")
     return result
    
def predictionBasedOnCom(event, context):
    
    print("************* ---predictionBasedOnCom--- ***************")
    # Importing data
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket='lambda.data.events', Key='lambda_event_data_comm.csv')
    contents = data['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(contents))
    # Preprocessing steps
    df.dropna(inplace=True)
    commData = event['commData']
    eventUserSourceDirect = commData["eventUserSourceDirect"]
    nbrOfDaysOfCommunicationPicsDirect = commData["nbrOfDaysOfCommunicationPicsDirect"]
    eventUserSourceSocial = commData["eventUserSourceSocial"]
    nbrOfDaysOfCommunicationPicsSocial = commData["nbrOfDaysOfCommunicationPicsSocial"]
    eventData = event['eventData']
    startDate = eventData["startDate"]
    startTime = datetime.datetime.strptime(startDate, "%Y-%m-%dT%H:%M:%S.%fZ").hour
    title = eventData["title"]
    titleIsQuestion = 0.2 if title.endswith('?') else 0.4
    if 'english' in title:
        titleLanguage = 2
    elif 'french' in title:
        titleLanguage = 3
    else:
        titleLanguage = 1
    title_length = len(title)
    title_length_squared = title_length * title_length
    description = eventData["description"]
    description_length = len(description)
    description_length_squared = description_length * description_length
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
    LE = LabelEncoder()
    columns_to_encode = ['start_time', 'title_is_question', 'title_language', 'platform', 'location', 'access', 
    'category', 'type', 'isFree', 'requiredSignIn', 'user_device', 'user_os', 'user_country','Tag1','Tag2','Tag3','Tag4']
    for column in columns_to_encode:
        df[column] = LE.fit_transform(df[column])
    features = df.drop(['event_registered','event_visitors','event_conversionRate'],axis=1)
    label = df.event_registered
    X_train, X_test, y_train, y_test = train_test_split(features,label, test_size=0.2, random_state=0)
    sc = MinMaxScaler()
    features = sc.fit_transform(features)
    features = df.drop(['event_registered','event_visitors','event_conversionRate'],axis=1)
    label = df.event_registered
    X_train, X_test, y_train, y_test = train_test_split(features,label, test_size=0.2, random_state=0) 
    variables_to_encode = [ address, access, category, type]
    mappingType = {"virtual": 3, "physical": 1, "hybrid": 2}
    encoded_type = mappingType[type]
    if address == 'Evey virtual venue':
        address_int = 2
    else:
        address_int = 1
    mappingAccess = {"public": 3, "private": 1, "restricted": 2}
    encoded_access = mappingAccess[access]
    mappingCategory = {"event": 7, "conference": 6, "fair": 5, "training": 4, "bootcamp": 3, "show": 2, "other": 1}
    encoded_category = mappingCategory[category]
    encoded_isFree = 2 if isFree else 1
    encoded_requiredSignIn = 0.5 if requiredSignIn else 1
    if speakerNbrOfParticipants == 0:
        speakerNbrOfParticipants = 20
    Xnew = [[1, startTime, title_length, title_length_squared, titleIsQuestion, titleLanguage, description_length,description_length_squared, 2, 
    address_int, encoded_access*12, encoded_category, encoded_type, encoded_isFree, nbrOfRegistrationFields, encoded_requiredSignIn,
    nbrOfSpeakers, speakerNbrOfEvents, speakerNbrOfParticipants, nbrOfSponsors, sponsorNbrOfEvents, sponsorNbrOfParticipants*0.2, 2, 1, 1, 25, 1,
    eventUserSourceDirect, nbrOfDaysOfCommunicationPicsDirect, eventUserSourceSocial,nbrOfDaysOfCommunicationPicsSocial, 10, 10, 1, 1, 1, 1]]
    RF = RandomForestRegressor()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    ynew = RF.predict(Xnew)
    prediction = max(5, ynew[0])
    prediction_weighted = prediction * 0.5
    # Calcul de l'intervalle de confiance
    tolerance = 0.30 * prediction_weighted
    lower_bound = max(prediction_weighted - tolerance, 5)
    upper_bound = prediction_weighted*3
    result = {
       "prediction": prediction_weighted,
       "lower_bound": lower_bound,
       "upper_bound": upper_bound
    }
    print("Random Forest Regressor")
    print("------------------------")
    print('Score Train : ',"{:.2%}".format(RF.score(X_train, y_train)))
    print('Score Test  : ',"{:.2%}".format(RF.score(X_test, y_test)))
    print("X=%s, Predicted_Registered=%s" % (Xnew[0], ynew[0]))
    print("Intervalle de confiance : [{:.2f}, {:.2f}]".format(lower_bound, upper_bound))
    print("------------------------")
    return result