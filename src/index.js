const express = require('express')
const userRouter = require('./routers/user')
const associatesRouter = require('./routers/associates')

require('./db/mongoose')

const app = express()

app.use(express.json())
app.use(userRouter)
app.use(associatesRouter)

app.get('', (req, res) => {
    res.send('Hello express')
})

//app.com
//app.com/help
//app.com/about


app.listen(3000, () => {
    console.log('Server is up on port 3000')
})


//PYTHON

const {spawn} = require('child_process');
app.get('/oython', (req, res) => {
 
 var dataToSend;
 // spawn new child process to call the python script
 const python = spawn('python', ['hello.py']);
 // collect data from script
 python.stdout.on('data', function (data) {
  console.log('Pipe data from python script ...');
  dataToSend = data.toString();
 });
 // in close event we are sure that stream from child process is closed
 python.on('close', (code) => {
 console.log(`child process close all stdio with code ${code}`);
 // send data to browser
 res.send(dataToSend)
 });
 
})