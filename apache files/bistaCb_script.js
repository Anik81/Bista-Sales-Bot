(function(window, document) {


    // body.innerHTML(doc);
    //document.querySelector(`#form`)

    var chatElement = document.createElement('div');
    chatElement.className = "chat_window_parent";
    //elemDiv.style.cssText = 'position:absolute;width:100%;height:100%;opacity:0.3;z-index:100;background:#000;';
    chatElement.innerHTML =  
            `   
                <div id = "welcome_container">
                    <h1>Welcome to Bista Solutions Chatbot</h1>
                    <form id="bista_chat_bot_user_info_form">
                        <label for="email">Email:</label>
                        <input type="email" id="email" name="email" required>
                        
                        <label for="contact-number">Contact Number:</label>
                        <input type="text" id="contact-number" name="contact_number">
                        
                        <label for="address">Address:</label>
                        <textarea id="address" name="address"></textarea>
                        
                        <button type="submit">Submit</button>
                    </form>
                    <p id="error-message"></p>
                </div>

                <div id="chat-window">
                    <div id="chat_header"><span>Bista ChatBot </span> <span id="close_chat">x</span></div>
                    <ul id="chat-messages">
                        <li class="li_chat_from_user">
                            <div class="chat_from_user"></div>
                        </li>
                        <li class="li_chat_from_bot">
                            <div class="chat_from_bot"></div>
                        </li>
                    </ul>
                    <div id="chat_input">
                        <input type="text" id="user-input" placeholder="Type a message..." />
                        <button id="send">Send</button>
                    </div>
                    
                </div>
            `;

    document.body.appendChild(chatElement);

    document.getElementById("close_chat").addEventListener("click", closeChatBox);
    document.getElementById("chat_header").addEventListener("click", openChatBox);
    document.getElementById("send").addEventListener("click", sendMsg);

    var chat_window =document.getElementById("chat-window")
    var welcome_window =document.getElementById("welcome_container")
    chat_window.style.display = "none";

    function closeChatBox(event) {
        event.stopPropagation()
        document.getElementById("chat-messages").style.display = "none";
        document.getElementById("chat_input").style.display = "none";
    }

    function openChatBox() {
        document.getElementById("chat-messages").style.display = "block";
        document.getElementById("chat_input").style.display = "flex";
    }

    function sendMsg(){
        var msg =  document.getElementById("user-input").value;
        if(msg.trim()){
            var liElement = document.createElement('li');
            liElement.className = "li_chat_from_user";
            liElement.innerHTML = `<div class="chat_from_user">${msg}</div>`
            document.getElementById("chat-messages").appendChild(liElement)
            document.getElementById("user-input").value = ""


            var liElementFromBot = document.createElement('li');
            liElementFromBot.className = "li_chat_from_bot";

            let userEmail = localStorage.getItem('user_email');
            let sessionId = localStorage.getItem('session_id');

            fetch("http://localhost:5002/api/chatbot", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: msg, session_id: sessionId, email: userEmail })
            })
            .then(response => response.json())
	    .then(data => {
                console.log("Received response from API:", data);
                liElementFromBot.innerHTML = `<div class="chat_from_bot">${data.response}</div>`
                document.getElementById("chat-messages").appendChild(liElementFromBot)
            })
            .catch(error => {
                console.error("Error:", error);
            });
        }
        
    }

    document.getElementById('bista_chat_bot_user_info_form').addEventListener('submit', function(evt){
        evt.preventDefault();

        var email = document.getElementById('email').value
        var contactNumber = document.getElementById('contact-number').value
        var address = document.getElementById('address').value
        var error_section = document.getElementById('error-message')

        const apiUrl = 'http://localhost:5002/api/submit_info';
        const data = {
            email: email,
            contact_number: contactNumber,
            address: address
        };

        const requestOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        };

        fetch(apiUrl, requestOptions)
            .then(response => {
                //if (response.response === 'Email verified!! You can ask your questions now.') {
                if(response.status == 200){
                    // Store email and session ID in local storage
                    localStorage.setItem('user_email', email);
                    localStorage.setItem('session_id', response.session_id);
                    // Redirect to chat page
                    welcome_window.style.display = "none";
                    chat_window.style.display = "block";
                   
                } else {
                    // Display error message
                    error_section.innerHTML += response.message;
                }
            }).catch(error => {
                console.log('error',error)
                error_section.innerHTML += 'Error occurred while processing your request.'
            });
    })


})(window, document);
