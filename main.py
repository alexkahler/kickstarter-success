from model.keras_model import *

def main():
    # Get data:

    #shows distribution of failed/success for categories food, games and technology
    get_distribution()

    #Get all data for simple/alt model
    #X, Y = get_data()

    #Get "slim" data for simple/alt model
    X,Y = get_slim_data()

    #Get "slim" data for category model, that is balanced for binary classification
    # 1st argument is category you wish to test, 2nd argument is column to exclude from test
    #X,Y = get_balanced_slim_cat('category_food', 'usd_goal')

    #Runs dummy prediction for baseline accuracy
    dummy_cross_validation(X, Y, 5)

    # cross_validation(X, Y, k, model, epochs, batch_size)
    cross_validation(X, Y, 5, 'simple', 100, 64)


if __name__ == "__main__":
    main()
