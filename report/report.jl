### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a8c95550-65bb-11ed-0ff5-d16dfa289d86
begin
	using Markdown
	using InteractiveUtils
	using PlutoUI
end

# ╔═╡ ce9501b4-2755-4d48-8c19-41fe22216078
md"""
# Deep Learning - Neural Networks
## Aristotle Universtity Thessaloniki - School of Informatics
### Assignment 1: Multilayer Perceptron
#### Antoniou, Antonios - 9482
#### aantonii@ece.auth.gr
"""

# ╔═╡ ed01e67c-0da0-4da9-932e-12df24523f15
md"""
## Introduction

In the first assignment of the Deep Learning course, our goal is to make a Neural Network **(NN)**, of any architecture, built for classification of problems with many classes (i.e. not a binary classifier). The selected one was a Multilayer Perceptron **(MLP)**, that is the simplest and most easily digestible of all, so we can dive deeper into the mathematic side of the project and have time to develop it all from scratch.
\
Before we start dissecting the algorithm into its components, we need to make sure we reminded ourselves of the most important terms, briefly.

## Terminology recap

#### What is a Perceptron?

A Perceptron is an entity simulating the function of a **neuron** in the human brain. In that sense, it simply accumulates information from sources before it (for which it has different **"weights"**, or degrees of **significance**), it processes it (that is, applies a certain arithmetic operation on all the collected data), and passes the output to the neurons that are connected to it afterwards. We can look at a Perceptron like this:

$(LocalResource(
	"perceptron.jpg",
	:style => "width: 50%;
			   text-align: center;
			   display: block;
			   margin-left: auto;
			   margin-right: auto;"
))
"""

# ╔═╡ 6d0a0338-87d7-40db-9b0b-bcbc317b2386
md"""
We refer to $f(\cdot)$ as the **activation function** of the Perceptron. It is used to keep the output on a relatively predictable range and filter out insignificant inputs, but also filter in significant ones.
\
**Note**: Oftentimes, as well as on this project, the $x_{0}$ and $w_{j,0}$ parameters belong to the **bias**: An *offset* to the inputs that either trainable, or given explicitely by the designer.
"""

# ╔═╡ 848293d0-3e64-4782-ab26-dfc855870254
md"""
#### Multilayer perceptrons

MLP's are simply layers of Perceptrons connected with each other, to make a structure that's capable of solving problems, or approximating functions that are non-linear and complex in general. They are:
* Fully connected: All neurons in layer $i$ communicate with all neurons in layer $i+1$.
* Feedforward: Information, other than errors, travels to the next layers and never comes back.
We will delve into the details, by explaining the structure of the MLP factory that was developed for this report.
"""

# ╔═╡ a50be0ca-fa46-43cc-9a5e-6c2b3d7c8058
md"""
## Coding and training an MLP

We split the classifier into its basic entities, that we turned into Python classes. This admittedly made the program a little less efficient, and the parameter changes a little longer to write. However, efficient libraries for Deep Learning have already been written. The reasons behind that Object-Oriented approach are pretty simple. First, it's a habit! Secondly, it makes the algorithm written a lot more readable, easier to debug and fully understand, since that's the bottom line of the project in question.
\
Having gotten that out of the way, we will now briefly explain the functionality of each of the 3 classes that the classifier is comprised of.

##### Neuron

This is the modelling of the Perceptron, explained above. **It will also be reffered to as a *Node*,** like it belongs to a directed graph. For the Neuron to be capable of being trained there is a myriad of attributes and inbetween values to be kept and updated during each iteration of testing.
\
First and foremost, a Node needs to know if it belongs to the input, output or any hidden layer. In the first case, it only broadcasts the value given to it to every Node in the first hidden layer. In the latter case, it needs to keep track of the target value given to it, and compare it to the one that it output. Output Nodes calculate their error $e$, and $\delta$ value, that are kept as attributes:
```python
def get_error(self):
	return self._d - self._y
```
```python
delta = e * self.df(node.u)
```
As you can see, the Neuron has also stored a `callable` for activation function *`f`* and its derivative *`df`*, along with:
\
$y=f(u)$, for which
\
$u=\underline{w}\cdot\underline{x}$
\
The $\underline{w}$ vector is kept internally, for each Neuron, along with the weights before the current iteration of training, for reasons we will explore later on. For the sake of consistency, the rest of the class attributes will be listed here, and also explained later:
* ID's for the layer it lies in, and an its own ID inside the layer,
* The learning rate $\eta$
* The target $d$ (allocated and initialized in case of an output neuron)
* The number of inputs $n_{in}$
"""

# ╔═╡ b55ce313-94d6-4a68-8b13-c344b22fddba
md"""
##### Layer

The Layer class is a wrapper around the array of Nodes that make it up. It contains the number of Nodes of the current and the previous layer, essentially kept so that the data can be passed on to the constructor of each Node. It also contains the type $t$ of layer it is (once again, input, output, or hidden) and the layer ID it comes with.
\
All in all, its function is to simply organize Neurons into arrays, so they can process information both during predicting and training in a more easy-to-track way.

##### Network

It encapsulates all the attributes and routines an MLP needs to be constructed, trained, tested and used. Apart from the attributes engrained in the array of Layers (and subsequently the arrays of Nodes), it also has:
* The $\alpha$ momentum factor,
* The resulting accuracy,
* The total error $e^{'}$, and
* The array of the resulting $\delta$ values for each Neuron of the Network, for a single sample during training, $sdeltas$. 
The class implements the *`predict()`*, *`train()`* and *`test()`* functions.

###### Predict

Function signature:
```python
def predict(self, x)
```
We create an array $y=x$. For the first layer, we simply assign:
```python
for i in range(len(x)):
	self.layers[0].nodes[i].y = x[i]
```
For the rest of them, we use the same $y$ vector to place the result in. The output of layer $i$ is the input of layer $i+1$, until the output layer is reached. As usual, for each Neuron of the layer, $u$ is calculated first, then $y$ is extracted. The function returns the entirety of $\underline{y}$, so we can keep all the probability values that the output Neurons concluded in. If *`classes`* is the array of available classification options, this means that in the occassion that *`index_of(max(y)) = i`*, then **`classes[i]`** is the decision the Network has made.
###### Train
Function signature:
```python
def train(self, x, d, batch_size, epochs, minJ)
```
For reference, here is the whole implementation of the function (without the implementation of the auxiliary routines constructed to make it more readable):
```python=
for iter in range(epochs):
	perm = np.random.permutation(size)
	x, d = x[perm], d[perm]
	curr_idx = 0
	self.e = 0
	for b in range(0, size, batch_size):
		b_size = min(batch_size, size - b)
		for p in range(b_size):
			curr_x = x[curr_idx]
			curr_d = d[curr_idx]
			set_targets(out_layer, nout, curr_d)
			self.predict(curr_x)
			self.update_output_errors(nout, out_layer)
			self.update_hidden_deltas(out_layer, last_idx)
			curr_idx += 1
		last_hidden = self.layers[last_idx - 1]
		update_output_weights(last_hidden, out_layer, nout, self.eta, self.alpha)
		self.update_hidden_weights(last_hidden, out_layer, last_idx)
		self.reset_errors_and_deltas(nout, out_layer, last_idx)
```
The routine is given the inputs $x$, and target outputs $d$ of the dataset it will be trained on. We also supply:
* The *batch size*, which is the number of samples the Network makes a prediction and calculates $e$ and $\delta$ values for,
* The *epochs*, which is the number of iterations of training throughout all samples,
* The *minJ* parameter, which is the value of the *loss function* at which the Network stops training, because we deem it reached a level of correctness we are satisfied with.
**Remarks**:
1. As one could guess, the batch size is kept consistent until we've reached the last batch. In that case, the batch-specific size is either the same, or the remainder of the samples, after we've gone through the rest
2. We will break down the training procedure into epochs. Every epoch requires the same processes and calculations, so all we need is to describe the progression of the algorithm during just one.
3. During each epoch, the initial inputs and targets are shuffled using the same permutation of indices, so we can keep the Network from overfitting. 

When the Network is first made, the weights are randomly initialized. However, since symmetry has to be avoided, the weights are given by a random generator following a **uniform distribution between 0 and 1**.
\
For every sample $p$, the output nodes are informed about the target values they are supposed to output. The `predict()` function is called, and directly every output Node's error is calculated, together with $\delta$, like shown above. For the rest, hidden, nodes, we need to calculate $\delta$ once again, but differently this time.
\
Let's take Node $n_{i}^{(l)}$ as an example. The notation means we're dealing with the Node with $ID=i$, on layer $l$:
* $\delta=\dot{f}(u)\cdot\sum_{j=1}^{N(l+1)}(\delta_{j}^{l+1}\cdot w_{j,i})$
When it comes to the implementation in Python, that's what we need the $sdeltas$ array, that keeps track of $\delta$ each Neuron calculates, layer by layer, so that the error can be propagated to the previous layer, up until we've reached the first hidden nodes. That is **Back Propagation (BK)**:
```python
for j in range(next_layer.n):
	delta += next_layer_deltas[j] * next_layer.nodes[j].w[i + 1]
return dfu * delta
```
We use *`w[i+1]`*, since *`w[0]`* is **reserved** for the **bias** of each Neuron.
\
As can be seen in the last 3 lines of code, every time we've examined enough samples, it's time to update the weights of the Network, then reset all the values that relate to errors: $e$ and $\delta$ for all Nodes.
\
Regarding the weight update, once we've gone through *`b_size`* of samples, summing up errors and $\delta$'s we calculate for Neuron $i$, on layer $l$:
* $new\_w_{i,j}=w_{i,j}+\eta\cdot\delta_{i}^{l}+y_{j}^{l-1}+\alpha\cdot previous\_w_{i,j}$
This is what we need *`wprev`* for, the array of weights before the current iteration, that we talked about earlier:
```python
for j in range(node_i.n_in):
	wprev = node_i.wprev[j + 1]
	node_i.wprev[j + 1] = node_i.w[j + 1]
	node_i.w[j + 1] += eta * node_i.delta * last_hidden.nodes[j].y + alpha * wprev
```

###### Testing
	
Lastly, there is a decision to make on the division of the data. Generally speaking, we give 60% of the data to the Network to train and keep 40% for testing. We have to pay special attention to pre-processing: the training data is shuffled before every epoch, but we need to make sure we keep a balance between the samples of different classes the Network trains on. There is a significant possibility of **overfitting**, with a bias towards the class that had the most training samples.
\
Here is an example of how that was handled using the [Iris dataset](https://www.kaggle.com/datasets/uciml/iris?resource=download):
```python
samples, targets = split_into_classes(x, y, 3, species)
x_train, x_test, y_train, y_test = 
	split_trainset_testset(samples, targets, train_fraction, species)
```
Here, the *`split_into_classes`* function uses the *`species`* dictionary to split the classes of the dataset into different arrays inside *`samples`*.
\
Afterwards, *`split_trainset_testset`*, takes approximately a percentage of each sample class equal to *`train_fraction`* and places them all into *`x_train`* (and the respective targets into *`y_train`*). The rest goes into the tests.

##### Acknowledgement

Any of the proposed models seem to be performing sub-par for non-linearly separable datasets. The loss function is minimized, however it soon reaches a plateau and takes considerable time in order to "escape" that region. All in all, the MLP works pretty well with linearly separable classes, such as [Iris](https://www.kaggle.com/datasets/uciml/iris?resource=download).

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─a8c95550-65bb-11ed-0ff5-d16dfa289d86
# ╟─ce9501b4-2755-4d48-8c19-41fe22216078
# ╟─ed01e67c-0da0-4da9-932e-12df24523f15
# ╟─6d0a0338-87d7-40db-9b0b-bcbc317b2386
# ╟─848293d0-3e64-4782-ab26-dfc855870254
# ╟─a50be0ca-fa46-43cc-9a5e-6c2b3d7c8058
# ╠═b55ce313-94d6-4a68-8b13-c344b22fddba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
