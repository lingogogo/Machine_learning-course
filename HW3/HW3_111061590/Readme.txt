Function defined:
	BLR is Baysian linear regression.
	MLR is maximum likelihood regression.
	change_setonum is make the data transfer to number. (Not use in the HW3.py)
	drawlinearline can draw the line on the figure.
	basis_function_gau is the gaussian function.
	basis_function_sigmoid is the sigmoid function.
	design_matrix is defined on the book, and I just code it. (The default basis function is sigmoid function, because it has better performance.)
	main function can run whole neccessary data and steps.
	MLR_beta_intercept(X, y) and BLR_beta_intercept(X , y) can give custom slope and intercept with correct data input.
Manual:
Run directly, and you can get the mean square error for the two method and question 4 answer (all of the used linear model).

If you want to get the picture in the question 3. You must uncomment all the Q3 code and run.(Best fit line figure)
It take 20~30 minutes to run the bayesian intercept and slope.
The second picture use custom function to achieve the Q3 problem, and define two function for calculating the slope and intercept.(Best fit line figure)

I also output the test, train and validation data into excel for checking data.