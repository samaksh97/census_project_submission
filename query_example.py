import requests

url = "https://udacity-mlops-04-vela-7602020815db.herokuapp.com/"

test_conncetion = requests.get(url=url)

print("Test root endpoint")
print(test_conncetion.status_code)
print(test_conncetion.content.decode())


print("Test inference endpoint")
test_predition = requests.post(
    url=url + 'predict/',
    json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "hours-per-week": 40,
            "native-country": "United-States"}
)
print(test_predition.status_code)
print(test_predition.content.decode())
