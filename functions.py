from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def deal_with_missing_values(data):
    for x in data.columns:
        if data[x].dtype == "int64":
            data[x] = data[x].fillna(data[x].mean())
        elif data[x].dtype == "float64":
            data[x] = data[x].fillna(data[x].mean())

        elif data[x].dtype == "O":
            data = data.drop(x,axis=1)
    for x in data.columns:
        if data[x].isnull().any() == True:
            data = data.drop(x, axis=1)
    return data



def deal_with_missing_values(data):
    for x in data.columns:
        if data[x].dtype == "int64":
            data[x] = data[x].fillna(data[x].mean())
        elif data[x].dtype == "float64":
            data[x] = data[x].fillna(data[x].mean())

        elif data[x].dtype == "O":

            for x in data:
                if data[x].dtype == "O":
                    new = []
                    for y in data[x]:
                        if type(y) == float:
                            new.append("theunknown")
                        else:
                            new.append(y)
                    data[x] = new

    return data




def check_for_missing_values(data):
    i=0
    for x in data.isnull().any():
        if x == True:
            i+=1
    if i>0:
        return print("exei akoma missing values")
    else:
        return print("den exei missing values")



def check_to_mikos_apo_ta_columns(data):
    i=0
    for d in data:
        mikos = data[d].count()
        break
    for x in data:
        if data[x].count() < mikos:
            i+=1
    if i>0:
        return "den exoun to idio mikos ola ta columns"
    else:
        return "ola ta columns exoun to idio mikos"


def label_encoder(data):
    enc = LabelEncoder()
    for x in data:
        if data[x].dtype == "O":
            d = enc.fit_transform(data[x])
            data[x] = d
    return data

def standard_scaler(data):
    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(data)
    return data