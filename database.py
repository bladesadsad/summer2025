from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Enum as SQLEnum, DateTime, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime
from enum import Enum as PyEnum
from passlib.context import CryptContext
from typing import Generator, Optional, List
import os
from enum import Enum 

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    phone = Column(String(20))
    role = Column(String(20), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Relationships
    cart = relationship("Cart", back_populates="user", uselist=False, cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user")

    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    slug = Column(String(255), unique=True, index=True)
    description = Column(String(1000))
    price = Column(Float, nullable=False)
    discount_price = Column(Float)
    category = Column(SQLEnum(CategoryEnum), nullable=False)
    gender = Column(SQLEnum(GenderEnum), nullable=False)
    color = Column(String(50))
    material = Column(String(100))
    brand = Column(String(100))
    country = Column(String(50))
    sku = Column(String(50), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)

    images = relationship("ProductImage", back_populates="product", cascade="all, delete-orphan")
    sizes = relationship("ProductSize", back_populates="product", cascade="all, delete-orphan")
    cart_items = relationship("CartItem", back_populates="product")
    order_items = relationship("OrderItem", back_populates="product")

    __table_args__ = (
        CheckConstraint('price > 0', name='check_price_positive'),
        CheckConstraint('discount_price IS NULL OR discount_price > 0', name='check_discount_price_positive'),
    )

class ProductImage(Base):
    __tablename__ = 'product_images'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete="CASCADE"))
    image_url = Column(String(500), nullable=False)
    alt_text = Column(String(100))
    is_main = Column(Boolean, default=False)
    position = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="images")

class ProductSize(Base):
    __tablename__ = 'product_sizes'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete="CASCADE"))
    size = Column(String(10), nullable=False)
    quantity = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = relationship("Product", back_populates="sizes")
    cart_items = relationship("CartItem", back_populates="size")
    order_items = relationship("OrderItem", back_populates="size")

    __table_args__ = (
        CheckConstraint('quantity >= 0', name='check_quantity_non_negative'),
    )

class Cart(Base):
    __tablename__ = 'carts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    session_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="cart")
    items = relationship("CartItem", back_populates="cart", cascade="all, delete-orphan")

    @property
    def total_items(self):
        return sum(item.quantity for item in self.items)

    @property
    def subtotal(self):
        return sum(item.total_price for item in self.items)
    
class CartItem(Base):
    __tablename__ = 'cart_items'
    
    id = Column(Integer, primary_key=True)
    cart_id = Column(Integer, ForeignKey('carts.id', ondelete="CASCADE"))
    product_id = Column(Integer, ForeignKey('products.id'))
    size_id = Column(Integer, ForeignKey('product_sizes.id'))
    quantity = Column(Integer, default=1)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    cart = relationship("Cart", back_populates="items")
    product = relationship("Product", back_populates="cart_items")
    size = relationship("ProductSize", back_populates="cart_items")

    @property
    def total_price(self):
        if self.product.discount_price:
            return self.product.discount_price * self.quantity
        return self.product.price * self.quantity

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    total_amount = Column(Float, nullable=False)
    status = Column(String(20), default="pending")  
    payment_method = Column(String(50))
    payment_status = Column(String(20), default="unpaid") 
    shipping_address = Column(String(500))
    billing_address = Column(String(500))
    tracking_number = Column(String(100))
    notes = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id', ondelete="CASCADE"))
    product_id = Column(Integer, ForeignKey('products.id'))
    size_id = Column(Integer, ForeignKey('product_sizes.id'))
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    discount_price = Column(Float)
    
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")
    size = relationship("ProductSize", back_populates="order_items")

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shop.db")
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
    echo=True  
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    
    with SessionLocal() as db:
        if db.query(User).count() == 0:
            admin = User(
                email="admin@example.com",
                hashed_password=pwd_context.hash("admin123"),
                full_name="Admin",
                phone="+1234567890",
                role="admin",
                is_verified=True
            )
            db.add(admin)
            
            sample_product = Product(
                name="Пример товара",
                slug="sample-product",
                description="Это пример товара для тестирования",
                price=1000.0,
                discount_price=800.0,
                category=CategoryEnum.MEN,
                gender=GenderEnum.MALE,
                material="Хлопок",
                brand="SampleBrand",
                country="Россия",
                sku="SP001",
                is_featured=True
            )
            
            sample_size = ProductSize(
                size="S",
                quantity=10,
                product=sample_product
            )
            
            sample_image = ProductImage(
                image_url="https://example.com/sample.jpg",
                alt_text="Пример товара",
                is_main=True,
                product=sample_product
            )
            
            db.add_all([sample_product, sample_size, sample_image])
            db.commit()
            print("База данных инициализирована")
            print("Создан администратор по умолчанию: admin@example.com / admin123")
            print("Создан тестовый товар")

if __name__ == "__main__":
    init_db()