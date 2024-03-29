# Option-Pricing-Deep-Learning-Algos
	Hutchinson et al. (1994) explored using neural nets to learn the famous Black 
	and Scholes (1973) option pricing formula, also developed and analyzed by Merton (1973).
	Using limited computing power and relatively small neural networks, they were able
	to show remarkably good performance in mimicking (learning) the following equation
	from simulated data.

	In order to create data for the assessment of how a deep neural net would learn
	this equation, a range of call option prices are simulated using a range of parameters
	shown below:

	-----------------------------------------------
	Parameter 				Range
	-----------------------------------------------
	Stock price (S) 		$10 – $500
	Strike price (K) 		$7 – $650
	Maturity (T) 			1 day to 3 years
	Dividend rate (q) 		0% – 3%
	Risk free rate (r) 		1% – 3%
	Volatility (σ) 			5% – 90%

	This data is divided into two random sets, one for training, comprising 240,000
	option prices, and are held out the remaining 60,000 prices for validation.
	Before passing the prices to the deep learning net, as we knew a facet of the 
	BlackScholes call option function, i.e., that the pricing function is linear homogenous in
	(S, K), i.e., C(S, K) = K · C(S/K, 1). Therefore,
	C(S, K)/K = C(S/K, 1)

	Accordingly, our data can be  modified by dividing both stock price S and call price C by
	strike price K. This normalized data was then fed into the deep learning net to fit
	the input variables S, K, T, q, r, σ (the feature set) to the output prices C.
	The details of the deep learning net are as follows. The size of the input is 6
	parameters. These are passed through 4 hidden layers of 100 neurons each. The
	neurons at each layer are chosen based on different “activation” functions that are
	respectively the following: LeakyReLU, ELU, ReLU, ELU. The final output layer
	comprises a single output neuron which can be set to be the standard exponential function
	exp(·) because the output of the neural net should be non-negative with certainty,
	as option prices cannot take negative values.

	the choice of “hyper-parameters” for the deep learning net is simple. At each
	hidden layer a dropout rate of 25% is set, so as to ameliorate overfitting. The loss
	function used for optimization is mean-squared error (MSE) and a batch size of 64, 
	with 10 epochs is used. The entire exercise results in fitting a total of
	31,101 coefficients (weights) for the deep learning model. The model was trained
	using Google’s TensorFlow package. 
