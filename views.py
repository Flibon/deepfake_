from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

# Home view - requires login
@login_required
def home(request):
    return render(request, "home.html", {})

# Signup view
def authView(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("base:login")  # Redirect to login page after signup
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})  # Use signup.html template
