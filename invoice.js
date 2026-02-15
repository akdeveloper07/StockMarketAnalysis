let order = {
 id:"GG1023",
 date:new Date().toLocaleDateString(),
 name:"Akash",
 total:500
};

localStorage.setItem("order", JSON.stringify(order));
window.location.href="404.html";
