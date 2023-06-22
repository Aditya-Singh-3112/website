import React from 'react';
import TextToImageForm from './TextToImageForm';
import axios from 'axios';

const App = () => {
  return (
    <div className="App">
      <TextToImageForm />
    </div>
  );
};

export default App;

const callTxtToImgAPI = async () => {
  try {
    const response = await axios.post('/api/txt_to_img_api', {
      prompt: 'Your prompt',
      n_images: 1,
      neg_prompt: 'Your negative prompt',
      guidance: 0.5,
      steps: 10,
      width: 800,
      height: 600,
      seed: 1234
    });

    console.log(response.data); // Handle the API response data here
  } catch (error) {
    console.error('Error:', error);
  }
};
