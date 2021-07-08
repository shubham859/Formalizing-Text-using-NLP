import streamlit as st
import joblib
import tensorflow as tf

# Defining Model Parameters
MAX_LEN = 200
UNITS = 256

# Loading tokenizers
tknizer_formal = joblib.load('tknizer_formal.pkl')
tknizer_informal = joblib.load('tknizer_informal.pkl')

# Defining Custom Loss Function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

@tf.function
def loss_function(real, pred):
    # Custom loss function that will not consider the loss for padded zeros.
    # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention
    # optimizer = tf.keras.optimizers.Adam()
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Loading trained model
model = tf.keras.models.load_model('Model_General', custom_objects={"loss_function": loss_function})

def predict(input_sentence):
    '''
    Takes input sentence and model instance as inputs and predicts the output.
    The prediction is done by using following steps:
    Step A. Given input sentence, preprocess the punctuations, convert the sentence into integers using tokenizer used earlier.
    Step B. Pass the input_sequence to encoder. we get encoder_outputs, last time step hidden and cell state
    Step C. Initialize index of '<' as input to decoder. and encoder final states as input_states to decoder
    Step D. Till we reach max_length of decoder or till the model predicted word '>':
            pass the inputs to timestep decoder at each timestep, update the hidden states and get the output token
    Step E. Return the predicted sentence.
    '''
    # Tokenizing and Padding the sentence
    inputs = [tknizer_informal.word_index.get(i, 0) for i in input_sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = MAX_LEN, padding = 'post')
    inputs = tf.convert_to_tensor(inputs)
    # Initializing result string and hidden states
    result = ''
    hidden = tf.zeros([1, UNITS]), tf.zeros([1, UNITS])
    # Getting Encoder outputs
    enc_out, state_h, state_c = model.encoder([inputs, hidden])
    dec_hidden = [state_h, state_c]
    dec_input = tf.expand_dims([tknizer_formal.word_index['<']], 0)
    # Running loop until max length or the prediction is '>' token
    for t in range(MAX_LEN):
        # Getting Decoder outputs fot timestep t
        output, state_h, state_c = model.decoder.timestepdecoder([dec_input, enc_out, state_h, state_c])
        # Getting token index having highest probability
        predicted_id = tf.argmax(output[0]).numpy()
        # Getting output token
        if tknizer_formal.index_word.get(predicted_id, '') == '>':
            break
        else:
            result += tknizer_formal.index_word.get(predicted_id, '')
            dec_input = tf.expand_dims([predicted_id], 0)
    # Postprocessing the result string to remove spaces between punctuations
    return result


  

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:#7ff4c7;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Formalizing Informal Text using Natural Language Processing</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Informal = st.text_input("Enter Text to be Formalized:")
    result = ""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = predict(Informal) 
        st.success('Formalized Text is: {}'.format(result))
     
if __name__=='__main__': 
    main()
