const { Router } = require('express')
const router = Router()

const { renderUploadForm, uploadFile } = require('../controllers/upload.controller')

router.get('/upload', renderUploadForm)

router.post('/upload', uploadFile)

module.exports = router