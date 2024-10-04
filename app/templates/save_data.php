<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $content = $_POST['content'];
    
    // Sanitize input to avoid issues
    $content = htmlspecialchars($content, ENT_QUOTES, 'UTF-8');
    
    // Append the content to a text file
    file_put_contents('data.txt', $content . PHP_EOL, FILE_APPEND | LOCK_EX);
    
    echo "Data saved successfully!";
} else {
    echo "Invalid request.";
}
?>