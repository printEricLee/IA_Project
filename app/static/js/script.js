function openMenu() {
    document.getElementById("menu").style.width = "250px";
}

function closeMenu() {
    document.getElementById("menu").style.width = "0";
}

//##############################################################################################

// Function to hide the loading message
function hide_word() {
    const loadingMessage = document.getElementById('loading-message');
    loadingMessage.style.display = 'none';
}

function show_word() {
    const loadingShowMessage = document.getElementById('loading-show-message');
    loadingShowMessage.style.display = 'flex';
}


