const { Router } = require('express')
const router = Router()

const { renderUploadForm, uploadFile } = require('../controllers/upload.controller')
const { isAdmin } = require('../helpers/auth')

router.get('/upload', isAdmin, renderUploadForm)

router.post('/upload', isAdmin, uploadFile)

module.exports = router