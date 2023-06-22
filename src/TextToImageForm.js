import React, { useState } from 'react';
import axios from 'axios';

const TextToImageForm = () => {
  const [prompt, setPrompt] = useState('');
  const [nImages, setNImages] = useState(1);
  const [negPrompt, setNegPrompt] = useState('');
  const [guidance, setGuidance] = useState(1.0);
  const [steps, setSteps] = useState(25);
  const [width, setWidth] = useState(256);
  const [height, setHeight] = useState(256);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('/txt-to-img', {
        prompt,
        n_images: nImages,
        neg_prompt: negPrompt,
        guidance,
        steps,
        width,
        height,
        seed
      });
      setResult(response.data.result);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h2>Text to Image Conversion</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor="prompt">Prompt:</label>
        <input type="text" id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} required />

        <label htmlFor="negprompt">Negative Prompt:</label>
        <input type="text" id="negprompt" value={negPrompt} onChange={(e) => setNegPrompt(e.target.value)} required />

        {/* Add other input fields for nImages, guidance, steps, width, height, and seed */}
        
        <button type="submit">Convert</button>
      </form>
      {result && (
        <div>
          <h3>Result:</h3>
          {/* Render the result */}
        </div>
      )}
    </div>
  );
};

export default TextToImageForm;
