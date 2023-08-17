# Email_model

We receive a lot of emails in our daily study and work. In addition to emails related 
to study and work, we also receive a lot of spam, including advertising emails, 
fraudulent emails, and so on. This task model judges whether the email is normal 
(ham) or spam email (spam) according to the text content contained in the email, to 
realize automatic spam filtering

#
The project is built on:  
*Python 3.8*   
*PyCharm*   

The processed data set are in the **/data** folder.   
The front-end website page code is in the **/templates/Stack** folder.   
The design files are in the **/documents** folder.    
The model codes are in the **/Email_model** folder; The **app.py** file is to start the project application; the **evluate.py** file is used to evaluate the data set and models. **Hybrid_model.py** file is implemented for the hybrid model; And **model.py** has three models inside, including Naïve Bayes, random forest and SVM model.    
The original Enron Email Dataset is available from: https://www.cs.cmu.edu/~./enron/   

* Download the project files from GitHub.
*	Install the Python environment. 
*	Move to the location of the file and open the terminal.
*	Run the server on terminal:
python app.py
*	Open the web browser by the link: http://127.0.0.1:5000 

Now, the project models are started and run on the local server. The user can input any email content on the front-end page to test if it is spam. And also, if users want to re-train the model, they can modify the data set and model codes, and next run the **model.py** and **hybrid_model.py** files to train the model again. After training, it will output the new model’s evaluation results and new models can be accessed on the same URL address to use it on the front-end page.
