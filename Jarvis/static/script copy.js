
// script.js

const chatInput = document.querySelector('.chat-input textarea');
const sendChatBtn = document.querySelector('.chat-input button');
const clearBtn = document.getElementById('clearBTN');
const stopAudioBTN = document.getElementById('stopAudioBTN');
const speechInputBTN = document.getElementById('speechInputBTN');

const chatbox = document.querySelector(".chatbox");


let userMessage;


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

            const text = await response.text();
            // messageElement.textContent = data;
            messageElement.innerHTML += text;
            chatbox.scrollTo(0, chatbox.scrollHeight)


};


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

const speechInput = async () => {


    userMessage = chatInput.value.trim();
    if (!userMessage) {
        return;
    }
    chatbox
    .appendChild(createChatLi(userMessage, "chat-outgoing"));
    chatbox
    .scrollTo(0, chatbox.scrollHeight);

    setTimeout(async () => {
        const incomingChatLi = createChatLi("Thinking...", "chat-incoming")
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        const messageElement = incomingChatLi


        const response = await fetch('/speechInput', {
            method: 'Get',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        const text = await response.text();
        // messageElement.textContent = data;
        messageElement.innerHTML += text;
        chatbox.scrollTo(0, chatbox.scrollHeight)



        
        // generateResponse(incomingChatLi);
    }, 600);


   
}
sendChatBtn.addEventListener("click", handleChat);
clearBtn.addEventListener("click", clearMemory);
stopAudioBTN.addEventListener("click", stopAudio);
speechInputBTN.addEventListener("click", speechInput);




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

        // const chatContainer = document.getElementById('chat-container');
        // const chatForm = document.getElementById('chat-form');
        // const messageInput = document.getElementById('message');

        // chatForm.addEventListener('submit', async (event) => {
        //     event.preventDefault();

        //     const message = messageInput.value;
        //     if (!message) return;

        //     messageInput.value = '';

        //     const response = await fetch('/chat', {
        //         method: 'POST',
        //         headers: {
        //             'Content-Type': 'application/x-www-form-urlencoded'
        //         },
        //         body: new URLSearchParams({ message })
        //     });

        //     const text = await response.text();
        //     chatContainer.innerHTML += text;
        // });