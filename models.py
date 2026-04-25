from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    cover_photo_id = Column(Integer, ForeignKey("photos.id", ondelete="SET NULL"), nullable=True)

    photos = relationship(
        "Photo",
        back_populates="collection",
        foreign_keys="Photo.collection_id",
        lazy="select",
    )
    cover_photo = relationship(
        "Photo",
        foreign_keys=[cover_photo_id],
        post_update=True,
    )


class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    stored_name = Column(String, unique=True, nullable=False)
    embedding = Column(String, nullable=True)        # JSON-encoded float list (CNN)
    color_histogram = Column(String, nullable=True)  # JSON-encoded float list (HSV hist)
    collection_id = Column(Integer, ForeignKey("collections.id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    collection = relationship(
        "Collection",
        back_populates="photos",
        foreign_keys=[collection_id],
    )
