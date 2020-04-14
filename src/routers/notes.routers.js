const { Router } = require('express')
const router = Router()
const { isAuthenticated } = require('../helpers/auth')

const { 
    renderNoteForm, 
    createNewNote, 
    renderNotes, 
    renderEditForm, 
    updateNote, 
    deleteNote 
} = require('../controllers/notes.controller')

// New Notes
router.get('/notes/add', isAuthenticated, renderNoteForm)

router.post('/notes/add', isAuthenticated, createNewNote)

// Get Notes
router.get('/notes', isAuthenticated, renderNotes)

// Edit Notes
router.get('/notes/edit/:id', isAuthenticated, renderEditForm)
router.put('/notes/edit/:id', isAuthenticated, updateNote)

// Delete Notes
router.delete('/notes/delete/:id', isAuthenticated, deleteNote)

module.exports = router