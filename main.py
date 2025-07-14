from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, BackgroundTasks, APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from database import Base, engine, SessionLocal, User, Product, pwd_context, get_db, ProductImage, ProductSize, CategoryEnum, CartItem, Cart, GenderEnum
from schemas import UserCreate, UserResponse, Token, ProductCreate, ProductResponse
from typing import Optional, List
from sqlalchemy.orm import Session
import logging
from urllib.parse import quote, unquote
from pydantic import BaseModel
from enum import Enum
import json

router = APIRouter(prefix="/cart", tags=["cart"])
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class OAuth2PasswordBearerWithCookie(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        access_token = request.cookies.get("access_token")
        if access_token:
            return access_token
        return await super().__call__(request)
    
SECRET_KEY = os.getenv("SECRET_KEY", "secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearerWithCookie(tokenUrl="login")

class CartItemRequest(BaseModel):
    product_id: int
    size_id: int
    quantity: int

class CartResponse(BaseModel):
    status: str
    cart_count: int
    cart_item_id: Optional[int] = None

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    token = token.replace("Bearer ", "")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
    except JWTError:
        return None
    
    return db.query(User).filter(User.email == email).first()

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.middleware("http")
async def add_user_to_request(request: Request, call_next):
    db = SessionLocal()
    try:
        token = request.cookies.get("access_token")
        print(f"Middleware - token found: {token is not None}")  # Логируем
        
        user = None
        if token:
            token = token.replace("Bearer ", "")
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                email = payload.get("sub")
                print(f"Middleware - email from token: {email}")  # Логируем
                
                if email:
                    user = db.query(User).filter(User.email == email).first()
                    print(f"Middleware - user found: {user is not None}")  # Логируем
            except JWTError as e:
                print(f"Middleware - JWT error: {str(e)}")  # Логируем
        
        request.state.user = user
        print(f"Middleware - user set: {request.state.user is not None}")  # Логируем
    finally:
        db.close()
    
    response = await call_next(request)
    return response

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "user": request.state.user})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "user": request.state.user})

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    email: str = Form(...),
    full_name: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    db: Session = Depends(get_db)
):
    if password != password_confirm:
        return templates.TemplateResponse("register.html", 
            {"request": request, "error": "Пароли не совпадают"})
    
    hashed_password = get_password_hash(password)
    user = User(email=email, full_name=full_name, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    
    return RedirectResponse("/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "user": request.state.user})

@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", 
            {"request": request, "error": "Неверный email или пароль"})
    
    access_token = create_access_token(data={"sub": user.email})
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/logout", response_class=HTMLResponse)
async def logout():
    response = RedirectResponse("/")
    response.delete_cookie("access_token")
    return response

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    if not request.state.user:
        return RedirectResponse("/login")
    
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": request.state.user
    })

@app.post("/api/products/", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

@app.get("/api/products/", response_model=list[ProductResponse])
def read_products(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    products = db.query(Product).offset(skip).limit(limit).all()
    return products

def create_first_admin(db: Session):
    if db.query(User).count() == 0:
        admin = User(
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Admin",
            role="admin"
        )
        db.add(admin)
        db.commit()
        print("Создан администратор по умолчанию: admin@example.com / admin123")

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    create_first_admin(db)
    db.close()

async def get_current_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора"
        )
    return current_user

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request, db: Session = Depends(get_db)):
    if not request.state.user or request.state.user.role != "admin":
        return RedirectResponse("/")
    users = db.query(User).all()
    products = db.query(Product).all()
    return templates.TemplateResponse("admin.html", {
    "request": request,
    "users": users,
    "products": products    
    })

@app.post("/admin/users/{user_id}/update-role")
async def update_user_role(
    user_id: int,
    new_role: str = Form(...),
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    user.role = new_role
    db.commit()
    return RedirectResponse("/admin", status_code=303)

CATEGORIES = [
    {"value": "men", "label": "Мужское"},
    {"value": "women", "label": "Женское"},
    {"value": "kids", "label": "Детское"},
    {"value": "accessories", "label": "Аксессуары"},
    {"value": "unisex", "label": "Унисекс"}
]

GENDERS = [
    {"value": "male", "label": "Мужской"},
    {"value": "female", "label": "Женский"},
    {"value": "unisex", "label": "Унисекс"}
]

@app.get("/admin/add-product", response_class=HTMLResponse)
async def add_product_page(request: Request):
    if not request.state.user or request.state.user.role != "admin":
        return RedirectResponse("/")
    
    return templates.TemplateResponse("add_product.html", {
        "request": request,
        "categories": CATEGORIES,
        "genders": GENDERS
    })

@app.post("/admin/add-product")
async def add_product(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    discount_price: Optional[float] = Form(None),
    category: str = Form(...),
    gender: str = Form(...),
    material: str = Form(...),
    brand: str = Form(...),
    is_active: bool = Form(True),
    image_url: str = Form(...),
    sizes_json: str = Form(...),  
    db: Session = Depends(get_db)
):

    if not request.state.user or request.state.user.role != "admin":
        return RedirectResponse("/")
    
    try:
        category_enum = CategoryEnum(category)
        gender_enum = GenderEnum(gender)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Некорректное значение. Допустимые категории: {[e.value for e in CategoryEnum]}, допустимые полы: {[e.value for e in GenderEnum]}"
        )

    try:
        sizes_data = json.loads(sizes_json)
        if not isinstance(sizes_data, dict):
            raise ValueError("Данные размеров должны быть в формате JSON объекта")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Неверный формат данных размеров: {str(e)}"
        )

    product = Product(
        name=name,
        description=description,
        price=price,
        discount_price=discount_price,
        category=category_enum,
        gender=gender_enum,
        material=material,
        brand=brand,
        is_active=is_active,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(product)
    db.commit()
    db.refresh(product)

    if image_url:
        main_image = ProductImage(
            product_id=product.id,
            image_url=image_url,
            is_main=True,
            created_at=datetime.utcnow()
        )
        db.add(main_image)

    if not sizes_data:
        raise HTTPException(
            status_code=400,
            detail="Необходимо указать хотя бы один размер"
        )
    
    for size, quantity in sizes_data.items():
        if not size.strip():
            continue
            
        try:
            quantity = int(quantity)
            if quantity < 0:
                raise ValueError("Количество не может быть отрицательным")
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Неверное количество для размера {size}: {str(e)}"
            )
        
        product_size = ProductSize(
            product_id=product.id,
            size=size.strip(),
            quantity=quantity,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(product_size)
    
    db.commit()
    
    return RedirectResponse("/admin", status_code=303)

@app.get("/admin/products/{product_id}/edit", response_class=HTMLResponse)
async def edit_product_page(
    request: Request,
    product_id: int,
    db: Session = Depends(get_db)
):
    if not request.state.user or request.state.user.role != "admin":
        return RedirectResponse("/")
    
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Товар не найден")

    main_image = db.query(ProductImage).filter(
        ProductImage.product_id == product_id,
        ProductImage.is_main == True
    ).first()
    
    return templates.TemplateResponse("edit_product.html", {
        "request": request,
        "product": product,
        "main_image": main_image.image_url if main_image else "",
        "categories": CATEGORIES,
        "genders": GENDERS,
        "current_category": product.category.name.lower()
    })

@app.post("/admin/products/{product_id}/edit")
async def edit_product(
    request: Request,
    product_id: int,
    name: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    discount_price: float = Form(None),
    category: str = Form(...),
    gender: str = Form(...),
    material: str = Form(...),
    brand: str = Form(...),
    is_active: bool = Form(True),
    image_url: str = Form(...),
    db: Session = Depends(get_db)
):
    if not request.state.user or request.state.user.role != "admin":
        return RedirectResponse("/")
    
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Товар не найден")
    
    try:
        category_enum = CategoryEnum[category.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Неверная категория")

    product.name = name
    product.description = description
    product.price = price
    product.discount_price = discount_price
    product.category = category_enum
    product.gender = gender
    product.material = material
    product.brand = brand
    product.is_active = is_active
    
    # Обновляем основное изображение
    main_image = db.query(ProductImage).filter(
        ProductImage.product_id == product_id,
        ProductImage.is_main == True
    ).first()
    
    if main_image:
        main_image.image_url = image_url
    else:
        new_image = ProductImage(
            product_id=product_id,
            image_url=image_url,
            is_main=True
        )
        db.add(new_image)
    
    db.commit()
    return RedirectResponse("/admin", status_code=303)

# Просмотр товара
@app.get("/admin/products/{product_id}")
async def view_product(
    request: Request,
    product_id: int,
    db: Session = Depends(get_db)
):
    if not request.state.user or request.state.user.role != "admin":
        return RedirectResponse("/")
    
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Товар не найден")
    
    
    images = db.query(ProductImage).filter(
        ProductImage.product_id == product_id
    ).all()
    
   
    sizes = db.query(ProductSize).filter(
        ProductSize.product_id == product_id
    ).all()
    
    return templates.TemplateResponse("view_product.html", {
        "request": request,
        "product": product,
        "images": images,
        "sizes": sizes
    })

class CategoryEnum(str, Enum):
    MEN = "men"
    WOMEN = "women"
    KIDS = "kids"
    ACCESSORIES = "accessories"
    UNISEX = "unisex"

class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/shop", response_class=HTMLResponse)
async def shop_page(
    request: Request,
    category: Optional[str] = None,
    gender: Optional[str] = None,
    page: int = 1,
    per_page: int = 6,
    db: Session = Depends(get_db)
):
    try:
        valid_categories = [cat.value for cat in CategoryEnum]
        valid_genders = [gen.value for gen in GenderEnum]
        
        if category and category not in valid_categories:
            category = None
        if gender and gender not in valid_genders:
            gender = None

        query = db.query(Product).filter(Product.is_active == True)

        if category:
            query = query.filter(Product.category == category)
        if gender:
            query = query.filter(Product.gender == gender)

        total_items = query.count()
        total_pages = max(1, (total_items + per_page - 1) // per_page)
        products = query.offset((page - 1) * per_page).limit(per_page).all()
        
        product_list = []
        for product in products:
            main_image = db.query(ProductImage).filter(
                ProductImage.product_id == product.id,
                ProductImage.is_main == True
            ).first()
            final_price = product.discount_price if product.discount_price else product.price
            
            product_data = {
                "id": product.id,
                "name": product.name,
                "price": product.price,
                "discount_price": product.discount_price,
                "final_price": final_price,
                "category": product.category.value,
                "gender": product.gender.value,
                "brand": product.brand,
                "material": product.material,
                "color": product.color,
                "country": product.country,
                "sku": product.sku,
                "is_featured": product.is_featured,
                "image_url": main_image.image_url if main_image else None,
                "alt_text": main_image.alt_text if main_image else product.name,
                "has_discount": product.discount_price is not None
            }
            product_list.append(product_data)

        def get_page_range():
            start = max(1, page - 2)
            end = min(total_pages, page + 2)
            return range(start, end + 1)

        context = {
            "request": request,
            "products": product_list,
            "categories": [{"name": cat.name, "value": cat.value} for cat in CategoryEnum],
            "genders": [{"name": gen.name, "value": gen.value} for gen in GenderEnum],
            "current_category": category,
            "current_gender": gender,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages,
                "prev_num": page - 1 if page > 1 else None,
                "next_num": page + 1 if page < total_pages else None,
                "page_range": list(get_page_range())
            }
        }
        
        return templates.TemplateResponse("shop.html", context)
        
    except Exception as e:
        print(f"Error in shop_page: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Произошла ошибка при загрузке страницы магазина"
        )

@app.get("/product/{product_id}", response_class=HTMLResponse)
async def product_detail(
    request: Request,
    product_id: int,
    db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product or not product.is_active:
        raise HTTPException(status_code=404, detail="Товар не найден")
    images = db.query(ProductImage).filter(
        ProductImage.product_id == product_id
    ).all()
    sizes = db.query(ProductSize).filter(
        ProductSize.product_id == product_id,
        ProductSize.quantity > 0
    ).all()
    
    return templates.TemplateResponse("product_detail.html", {
        "request": request,
        "product": product,
        "images": images,
        "sizes": sizes,
        "main_image": next((img for img in images if img.is_main), None)
    })

logger = logging.getLogger(__name__)

class CartService:
    @staticmethod
    def get_user_cart(db: Session, user_id: int) -> Optional[Cart]:
        return db.query(Cart).filter(Cart.user_id == user_id).first()

    @staticmethod
    def create_cart(db: Session, user_id: int) -> Cart:
        cart = Cart(user_id=user_id)
        db.add(cart)
        db.commit()
        db.refresh(cart)
        return cart

    @staticmethod
    def get_cart_items(db: Session, cart_id: int) -> List:
        return db.query(CartItem, Product, ProductSize)\
            .join(Product, CartItem.product_id == Product.id)\
            .join(ProductSize, CartItem.size_id == ProductSize.id)\
            .filter(
                CartItem.cart_id == cart_id,
                Product.is_active == True
            )\
            .all()

    @staticmethod
    def calculate_total(cart_items: List) -> float:
        return sum(
            (item.Product.discount_price or item.Product.price) * item.CartItem.quantity
            for item in cart_items
        )

@router.get("/", response_class=HTMLResponse)
async def view_cart(request: Request, db: Session = Depends(get_db)):
    try:
        if not request.state.user:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        cart = db.query(Cart).filter(Cart.user_id == request.state.user.id).first()
        
        cart_items = []
        total = 0.0
        
        if cart:
            items = db.query(CartItem, Product, ProductSize)\
                .join(Product, CartItem.product_id == Product.id)\
                .join(ProductSize, CartItem.size_id == ProductSize.id)\
                .filter(CartItem.cart_id == cart.id)\
                .all()

            for item in items:
                cart_item, product, size = item
                images = db.query(ProductImage).filter(
                    ProductImage.product_id == product.id
                ).all()
                
                price = product.discount_price if product.discount_price else product.price
                item_total = price * cart_item.quantity
                total += item_total
                
                cart_items.append({
                    'id': cart_item.id,
                    'product': {
                        'id': product.id,
                        'name': product.name,
                        'brand': product.brand,
                        'price': product.price,
                        'discount_price': product.discount_price,
                        'images': images
                    },
                    'size': {
                        'id': size.id,
                        'size': size.size,
                        'quantity': size.quantity
                    },
                    'quantity': cart_item.quantity,
                    'item_total': item_total
                })
        message = request.cookies.get("cart_message")
        error = request.cookies.get("cart_error")

        response = templates.TemplateResponse("cart.html", {
            "request": request,
            "cart_items": cart_items,
            "total": total,
            "message": message,
            "error": error
        })

        if message:
            response.delete_cookie("cart_message")
        if error:
            response.delete_cookie("cart_error")

        return response

    except Exception as e:
        logger.error(f"Ошибка при загрузке корзины: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Произошла ошибка при загрузке корзины"
        )
    

@router.post("/add", response_model=CartResponse)
async def api_add_to_cart(
    item_data: CartItemRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        if not request.state.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Требуется авторизация"
            )

        product = db.query(Product).filter(
            Product.id == item_data.product_id,
            Product.is_active == True
        ).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Товар не найден"
            )

        size = db.query(ProductSize).filter(
            ProductSize.id == item_data.size_id,
            ProductSize.product_id == item_data.product_id,
            ProductSize.quantity >= item_data.quantity
        ).first()
        
        if not size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Недостаточное количество или размер недоступен"
            )

        cart = CartService.get_user_cart(db, request.state.user.id)
        if not cart:
            cart = CartService.create_cart(db, request.state.user.id)

        existing_item = db.query(CartItem).filter(
            CartItem.cart_id == cart.id,
            CartItem.product_id == item_data.product_id,
            CartItem.size_id == item_data.size_id
        ).first()

        if existing_item:
            new_quantity = existing_item.quantity + item_data.quantity
            if new_quantity > size.quantity:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Максимально доступное количество: {size.quantity}"
                )
            existing_item.quantity = new_quantity
            item_id = existing_item.id
        else:
            cart_item = CartItem(
                cart_id=cart.id,
                product_id=item_data.product_id,
                size_id=item_data.size_id,
                quantity=item_data.quantity
            )
            db.add(cart_item)
            db.commit()
            item_id = cart_item.id

        cart_count = db.query(CartItem).filter(CartItem.cart_id == cart.id).count()
        
        return {
            "status": "success",
            "cart_count": cart_count,
            "cart_item_id": item_id
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при добавлении в корзину: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при добавлении товара"
        )

@router.get("/view", response_class=HTMLResponse)
async def view_cart(request: Request, db: Session = Depends(get_db)):
    try:
        if not request.state.user:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

        cart = CartService.get_user_cart(db, request.state.user.id)
        cart_items = []
        total = 0.0
        
        if cart:
            cart_items = CartService.get_cart_items(db, cart.id)
            total = CartService.calculate_total(cart_items)

        message = unquote(request.cookies.get("cart_message", ""))
        error = unquote(request.cookies.get("cart_error", ""))

        response = templates.TemplateResponse("cart.html", {
            "request": request,
            "cart_items": cart_items,
            "total": total,
            "message": message or None,
            "error": error or None,
            "user": request.state.user
        })

        if message or error:
            response.delete_cookie("cart_message")
            response.delete_cookie("cart_error")

        return response

    except Exception as e:
        logger.error(f"Ошибка при загрузке корзины: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Произошла ошибка при загрузке корзины"
        )

@router.post("/add-item", response_class=RedirectResponse)
async def add_to_cart(
    request: Request,
    product_id: int = Form(...),
    size_id: int = Form(...),
    quantity: int = Form(1, gt=0),
    db: Session = Depends(get_db)
):
    try:
        if not request.state.user:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

        product = db.query(Product).filter(
            Product.id == product_id,
            Product.is_active == True
        ).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Товар не найден"
            )

        size = db.query(ProductSize).filter(
            ProductSize.id == size_id,
            ProductSize.product_id == product_id,
            ProductSize.quantity >= quantity
        ).first()
        
        if not size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Недостаточное количество или размер недоступен"
            )

        cart = CartService.get_user_cart(db, request.state.user.id)
        if not cart:
            cart = CartService.create_cart(db, request.state.user.id)

        existing_item = db.query(CartItem).filter(
            CartItem.cart_id == cart.id,
            CartItem.product_id == product_id,
            CartItem.size_id == size_id
        ).first()

        if existing_item:
            new_quantity = existing_item.quantity + quantity
            if new_quantity > size.quantity:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Максимально доступное количество: {size.quantity}"
                )
            existing_item.quantity = new_quantity
        else:
            cart_item = CartItem(
                cart_id=cart.id,
                product_id=product_id,
                size_id=size_id,
                quantity=quantity
            )
            db.add(cart_item)

        db.commit()

        response = RedirectResponse(
            url="/cart/view",
            status_code=status.HTTP_303_SEE_OTHER
        )
        response.set_cookie(
            key="cart_message",
            value=quote("Товар добавлен в корзину"),
            max_age=3,
            path="/"
        )
        return response

    except HTTPException as he:
        response = RedirectResponse(
            url=f"/product/{product_id}",
            status_code=status.HTTP_303_SEE_OTHER
        )
        response.set_cookie(
            key="cart_error",
            value=quote(he.detail),
            max_age=3,
            path="/"
        )
        return response
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при добавлении в корзину: {str(e)}", exc_info=True)
        
        response = RedirectResponse(
            url="/cart/view",
            status_code=status.HTTP_303_SEE_OTHER
        )
        response.set_cookie(
            key="cart_error",
            value=quote("Ошибка при добавлении товара"),
            max_age=3,
            path="/"
        )
        return response

@router.post("/update/{item_id}", response_class=RedirectResponse)
async def update_cart_item(
    item_id: int,
    quantity: int = Form(..., gt=0),
    db: Session = Depends(get_db),
    request: Request = None
):
    try:
        if not request.state.user:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

        cart_item = db.query(CartItem).join(Cart).filter(
            CartItem.id == item_id,
            Cart.user_id == request.state.user.id
        ).first()
        
        if not cart_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Товар не найден в корзине"
            )
        
        size = db.query(ProductSize).filter(
            ProductSize.id == cart_item.size_id,
            ProductSize.quantity >= quantity
        ).first()
        
        if not size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Недостаточное количество товара"
            )
        
        cart_item.quantity = quantity
        db.commit()
        
        return RedirectResponse(
            url="/cart/view",
            status_code=status.HTTP_303_SEE_OTHER
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при обновлении корзины: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при обновлении корзины"
        )

@router.post("/remove/{item_id}", response_class=RedirectResponse)
async def remove_cart_item(
    item_id: int,
    db: Session = Depends(get_db),
    request: Request = None
):
    try:
        if not request.state.user:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

        cart_item = db.query(CartItem).join(Cart).filter(
            CartItem.id == item_id,
            Cart.user_id == request.state.user.id
        ).first()
        
        if not cart_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Товар не найден в корзине"
            )
        
        db.delete(cart_item)
        db.commit()
        
        response = RedirectResponse(
            url="/cart/view",
            status_code=status.HTTP_303_SEE_OTHER
        )
        response.set_cookie(
            key="cart_message",
            value=quote("Товар удален из корзины"),
            max_age=3,
            path="/"
        )
        return response
        
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при удалении из корзины: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при удалении товара"
        )

app.include_router(router)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                email="admin@example.com",
                hashed_password=get_password_hash("admin123"),
                full_name="Admin",
                role="admin",
                is_verified=True
            )
            db.add(admin)
            db.commit()
            logger.info("Создан администратор по умолчанию: admin@example.com / admin123")
    finally:
        db.close()
