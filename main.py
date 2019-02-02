import numpy as np
import matplotlib.pyplot as plt
from utils import data_prep, calc_mse, cf_train, gd_train, contrast_mse
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    text_feature_num = 160
    is_add_features = 'Y'  # Change between Y or N to test
    is_no_text = 'N'  # change between Y or N, Y means no text words feature added
    [train_x, train_y, validate_x, validate_y, test_x, test_y] = data_prep(text_feature_num, is_add_features, is_no_text)



    cf_start = time.time()
    cf_w = cf_train(train_x, train_y)
    cf_end = time.time()
    cf_t = cf_end - cf_start
    cf_train_mse = calc_mse(cf_w, train_x, train_y)
    cf_validate_mse = calc_mse(cf_w, validate_x, validate_y)
    #cf_test_mse = calc_mse(cf_w, test_x, test_y)


    #theta is learning rate, beta is beta, th is criteria epsilon
    theta, beta, th = 5e-6, 1e-3, 1e-4
    #w0 = np.random.rand(train_x.shape[1], 1)
    w0 = np.zeros((train_x.shape[1], 1))
    gd_start = time.time()
    gd_w, hist_gd_mse = gd_train(train_x, train_y, validate_x, validate_y, theta, beta, th, w0)
    gd_end = time.time()
    gd_t = gd_end - gd_start
    gd_train_mse = calc_mse(gd_w, train_x, train_y)
    gd_validate_mse = calc_mse(gd_w, validate_x, validate_y)
    #gd_test_mse = calc_mse(gd_w, test_x, test_y)

    #contrast_mse1 = contrast_mse(validate_y)



    print("Add new features?:", is_add_features)
    print("Text feature number:", text_feature_num)
    print("No text?", is_no_text)


    print("Closed form MSE on training set:", cf_train_mse)
    print("closed form MSE on validation:", cf_validate_mse)
    #print("Closed form MSE on test set:", cf_test_mse)
    print("Closed form time:", cf_t)

    print("Gradient hyperparameter:", "theta:", theta, "\t beta", beta, "\t threshold", th)

    print("Gradient descent MSE on training:", gd_train_mse)
    print("Gradient descent MSE on validation:", gd_validate_mse)
    #print("Gradient descent MSE on test:", gd_test_mse)

    print("Gradient descent time:", gd_t)
#    print("ContrastMSE", contrast_mse1)

    #Plotting performance from GD
    plt.plot(list(range(1,len(hist_gd_mse)+1)), hist_gd_mse)
    plt.ylabel('Performance (MSE)')
    plt.xlabel('Descent cycles')
    plt.show()