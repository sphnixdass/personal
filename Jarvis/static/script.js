
// script.js

const chatInput = document.querySelector('.chat-input textarea');
const sendChatBtn = document.querySelector('.chat-input button');
const clearBtn = document.getElementById('clearBTN');
const stopAudioBTN = document.getElementById('stopAudioBTN');
const speechInputBTN = document.getElementById('speechInputBTN');

const chatbox = document.querySelector(".chatbox");


let userMessage;
let isListening = false; // To track whether the loop is running
let stopLoop = false;    // To signal when to stop the loop

//OpenAI Free APIKey

const createChatLi = (message, className) => {
  const chatLi = document.createElement("li");
  chatLi.classList.add("chat", className);
  let chatContent =
    className === "chat-outgoing" ? `<p>${message}</p>` : `<p>${message}</p>`;
  chatLi.innerHTML = chatContent;
  return chatLi;
}

const generateResponse = async (incomingChatLi) => {
  const API_URL = "http://192.168.0.20:3010/chat";
  const messageElement = incomingChatLi
    .querySelector("p");


  const response = await fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: new URLSearchParams({ userMessage })
  });

  text = await response.text();
  text = convertMarkdownToHTML(text);

  //   // messageElement.textContent = data;
  //   if (text.includes('```')) {
  //     const regex = /```(.*?)```/gs;
  //     let result;
  //     while ((result = regex.exec(text)) !== null) {
  //         const codeBlock = result[1];
  //         text = text.replace(result[0], `<pre class="code-block">${codeBlock}</pre>`);
  //     }
  // }

  messageElement.innerHTML = text;
  chatbox.scrollTo(0, chatbox.scrollHeight)


};

function convertMarkdownToHTML(text) {
  // Escape HTML special characters to prevent XSS vulnerabilities
  // const escapeHtml = (str) => str.replace(/&/g, '&amp;')
  //                                .replace(/</g, '&lt;')
  //                                .replace(/>/g, '&gt;');

  const escapeHtml = (str) => str.replace(/&/g, '&amp;');



  // Replace headings
  text = text.replace(/^# (.*)$/gm, '<h1>$1</h1>');
  text = text.replace(/^## (.*)$/gm, '<h2>$1</h2>');
  text = text.replace(/^### (.*)$/gm, '<h3>$1</h3>');

  text = text.replace(/^ # (.*)$/gm, '<h1>$1</h1>');
  text = text.replace(/^ ## (.*)$/gm, '<h2>$1</h2>');
  text = text.replace(/^ ### (.*)$/gm, '<h3>$1</h3>');
  // Handle unordered lists
  const listRegex = /^( *)(\*|\-|\+) (.*)$/mg;
  let match;
  let inList = false;

  while ((match = listRegex.exec(text)) !== null) {
    if (!inList) {
      text = text.replace(listRegex, '<ul>$&</ul>');
      inList = true;
    } else {
      text = text.replace(/\n/gm, '</li>\n<li>'); // Close and open new li elements
    }
  }

  if (inList) {
    text += '</li></ul>';
  }

  // Replace links
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

  // Replace images
  text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1">');

  // Handle inline code and code blocks
  // text = text.replace(/`(.*?)`/g, '<code>$1</code>');
  
  const codeBlockRegex = /```(.*?)```/gs;
  let result;

  while ((result = codeBlockRegex.exec(text)) !== null) {
    let codeBlock = escapeHtml(result[1]);
    
    // Handle headings within code blocks
    codeBlock = codeBlock.replace(/^# (.*)$/gm, '<h1>$1</h1>');
    codeBlock = codeBlock.replace(/^## (.*)$/gm, '<h2>$1</h2>');
    codeBlock = codeBlock.replace(/^### (.*)$/gm, '<h3>$1</h3>');

    codeBlock = codeBlock.replace(/^ # (.*)$/gm, '<h1>$1</h1>');
    codeBlock = codeBlock.replace(/^ ## (.*)$/gm, '<h2>$1</h2>');
    codeBlock = codeBlock.replace(/^ ### (.*)$/gm, '<h3>$1</h3>');

    text = text.replace(result[0], `<pre class="code-block"><code>${codeBlock}</code></pre>`);
  }

  // Handle bold and italic
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
  text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');             // Italic

  // Replace paragraphs (note: this must be last to avoid wrapping HTML tags)
  const paragraphRegex = /^(?!<)([^\n]+)$/gm;
  let paragraphMatch;

  while ((paragraphMatch = paragraphRegex.exec(text)) !== null) {
    text = text.replace(paragraphMatch[0], '<p>' +  escapeHtml(paragraphMatch[1]) + '</p>');
  }

  return text;
}



// function convertMarkdownToHTML(text) {
//   // Replace headings
//   text = text.replace(/^# (.*)$/gm, '<h1>$1</h1>');
//   text = text.replace(/^## (.*)$/gm, '<h2>$2</h2>');
//   text = text.replace(/^### (.*)$/gm, '<h3>$3</h3>');

//   // Replace bullet points
//   text = text.replace(/^(\*|\-|\+) (.*)$/gm, '<ul><li>$2</li></ul>');

//   // Replace paragraphs
//   text = text.replace(/^(?!<)(.*)$/gm, '<p>$1</p>');

//     // Handle bold and italic within the text
//     text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
//     text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');             // Italic

//   // Code blocks
//   const regex = /```(.*?)```/gs;
//   let result;
//   while ((result = regex.exec(text)) !== null) {
//     codeBlock = result[1];
//     codeBlock = codeBlock.replace("<h1>", "#");
//     codeBlock = codeBlock.replace("</h1>", "");

//     codeBlock = codeBlock.replace("<h2>", "##");
//     codeBlock = codeBlock.replace("</h2>", "");

//     codeBlock = codeBlock.replace("<h3>", "###");
//     codeBlock = codeBlock.replace("</h3>", "");

//     text = text.replace(result[0], `<pre class="code-block">${codeBlock}</pre>`);
//   }

//   return text;
// }

const handleChat = () => {
  userMessage = chatInput.value.trim();
  if (!userMessage) {
    return;
  }
  chatbox
    .appendChild(createChatLi(userMessage, "chat-outgoing"));
  chatbox
    .scrollTo(0, chatbox.scrollHeight);

  setTimeout(() => {
    const incomingChatLi = createChatLi("Thinking...", "chat-incoming")
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    generateResponse(incomingChatLi);
  }, 600);
}


const clearMemory = () => {
  chatbox.innerHTML = "";
  chatInput.value = "";
  document.getElementById("inputTextbox").value = "";

  fetch('/clear')
    .then(response => {
      if (!response.ok) {
        throw new Error(`Network response was not ok (status: ${response.status})`);
      }
      return response.text();
    })
    .then(data => {
      // Process the received data
      console.log(data);
      // Update the UI, display data, etc.
    })
    .catch(error => {
      console.error('There has been a problem with your fetch operation:', error);
    });



}

const stopAudio = () => {
  fetch('/stopAudio')
    .then(response => {
      if (!response.ok) {
        throw new Error(`Network response was not ok (status: ${response.status})`);
      }
      return response.text();
    })
    .then(data => {
      // Process the received data
      console.log(data);
      // Update the UI, display data, etc.
    })
    .catch(error => {
      console.error('There has been a problem with your fetch operation:', error);
    });
}


async function speechInput() {



  userMessage = chatInput.value.trim();
  if (!userMessage) {
    return;
  }
  
  return new Promise(async (resolve) => {

    

    const response = await fetch('/speechInput', {
      method: 'Get',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });

    const text = await response.text();
    // messageElement.textContent = data;
    if (text !== "") {

    
      const parts = text.split("<Dass>");

      // Store the parts in separate variables
      const firstHalf = parts[0];
      const secondHalf = parts[1];

    
    chatbox.appendChild(createChatLi(secondHalf, "chat-outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);
    const incomingChatLi = createChatLi("Thinking...", "chat-incoming")
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    const messageElement = incomingChatLi



      messageElement.innerHTML = firstHalf;
      chatbox.scrollTo(0, chatbox.scrollHeight)
      // Reset state and button text when the loop ends
    }
    resolve(); // Resolve the promise when done

  }, 600);

}

async function speachloop() {

  if (isListening) {
    // Stop the loop if it's currently running
    stopLoop = true; // Signal to stop the loop
    speechInputBTN.textContent = 'Stopped'; // Change button text
    isListening = false; // Update the state
  } else {
    // Start the loop if not running
    isListening = true; // Update the state
    speechInputBTN.textContent = 'Listening...'; // Change button text
    for (let i = 0; i < 100; i++) {
      console.log("Listening... from loop" + i);
      if (stopLoop) break; // Exit the loop if stopLoop is true
      await speechInput(); // Start the loop
    }
    speechInputBTN.textContent = 'Stopped';

  }

}


sendChatBtn.addEventListener("click", handleChat);
clearBtn.addEventListener("click", clearMemory);
stopAudioBTN.addEventListener("click", stopAudio);
speechInputBTN.addEventListener("click", speachloop);




function cancel() {
  let chatbotcomplete = document.querySelector(".chatBot");
  if (chatbotcomplete.style.display != 'none') {
    chatbotcomplete.style.display = "none";
    let lastMsg = document.createElement("p");
    lastMsg.textContent = 'Thanks for using our Chatbot!';
    lastMsg.classList.add('lastMessage');
    document.body.appendChild(lastMsg)
  }
}

